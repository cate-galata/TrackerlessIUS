#   CODE ADAPTED FROM: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/network_architecture/generic_UNet.py

from copy import deepcopy
from torch import nn
import torch
import math
import itertools
import numpy as np
import torch.nn.functional
from torch.distributions import Normal
from networks.blocks import BlockEncoder, AsFeatureMap_down, AsFeatureMap_up, BlockDecoder, BlockFinalImg, Upsample, BlockQ, InitWeights_He

_LOG_SQRT_2PI = math.log(math.sqrt(2 * math.pi))


class Gaussian:
    """Drop-in replacement for torch.distributions.Normal covering just the
    ops this model needs (.loc, .scale, .rsample(), .sample(), .log_prob()).

    torch.distributions.Normal does input validation and other
    python-level/data-dependent control flow on construction and on every
    method call, which causes torch.compile / Dynamo to graph-break. This
    class is pure tensor arithmetic so the whole encode/decode path can be
    captured in a single compiled graph.
    """
    __slots__ = ('loc', 'scale')

    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def rsample(self):
        return self.loc + self.scale * torch.randn_like(self.loc)

    # alias: no reparameterization vs. direct sampling distinction needed
    # here since inference always runs under no_grad/inference_mode anyway
    sample = rsample

    def log_prob(self, value):
        var = self.scale * self.scale
        return -((value - self.loc) ** 2) / (2 * var) - torch.log(self.scale) - _LOG_SQRT_2PI


def soft_clamp(x: torch.Tensor, v: int=10):
    return x.div(v).tanh_().mul(v)

def soft_clamp_img(x: torch.Tensor):
    return (x.div(5).tanh_() + 1 ) / 2 

class MHVAE2D(nn.Module):

    def __init__(
        self, 
        modalities, 
        base_num_features, 
        num_pool,
        num_feat_img=1,
        weightInitializer=None,
        original_shape=(160,192,144),
        max_features=64,
        with_residual=False,
        with_se=True,
        logger=None,
        nfeat_finalblock=16,
        nb_finalblocks=6,
        clamp_value=10.,
        last_act='tanh'):
        """

        """
        super(MHVAE2D, self).__init__()

        self.weightInitializer = weightInitializer
        self.max_features = max_features
        self.num_feat_img = num_feat_img
        self.logger = logger
        self.last_act = last_act
        self.modalities = modalities
        self.clamp_value = clamp_value

        min_shape = [int(k/(2**num_pool)) for k in original_shape]  
        down_strides = [(2,2)]*(num_pool)
        up_kernel = down_strides

        nfeat_input = 1 
        nfeat_output = base_num_features
        self.first_conv = dict()
        # Layers for the modality-specific embeddings
        for mod in modalities:
            self.first_conv[mod] = nn.Conv2d(
                        in_channels=nfeat_input, 
                        out_channels=nfeat_output,
                        kernel_size=3,
                        stride=1, 
                        padding=1,
                        bias=True)

        self.conv_blocks_context = {mod:[] for mod in modalities}
        self.td =  {mod:[] for mod in modalities}
        for mod in modalities:
            nfeat_output = base_num_features
            for d in range(num_pool):
                nfeat_input = nfeat_output
                nfeat_output = min(2 * nfeat_output, max_features)
                self.conv_blocks_context[mod].append(
                    BlockEncoder(
                        nfeat_input, 
                        residual=with_residual, 
                        with_se=with_se)
                    )
                self.td[mod].append(
                    nn.Conv2d(
                        in_channels=nfeat_input, 
                        out_channels=nfeat_output, 
                        kernel_size=3, 
                        stride=down_strides[d], 
                        padding=1)
                    )
            
            self.conv_blocks_context[mod].append(
                BlockEncoder(
                    nfeat_output, 
                    residual=with_residual, 
                    with_se=with_se)
                )


        # Going from a feature volume to a feature vector (e.g (3,3,3)-->(1))
        self.bottleneck_down = AsFeatureMap_down(
            input_shape=[nfeat_output,]+min_shape, 
            target_dim=4*max_features
            )
        # Going from a feature vector to a feature volume (e.g. (1)-->(3,3,3)) 
        self.bottleneck_up = AsFeatureMap_up(
            input_dim=2*max_features, 
            target_shape=[max_features,]+min_shape)

        # Layers for the approximate posterior + prior
        nfeat_latent = max_features 
        self.tu = []
        self.conv_blocks_localization = []
        self.qz = []
        self.pz = []
        for u in np.arange(num_pool)[::-1]:
            nfeatures_from_skip = self.conv_blocks_context[modalities[0]][u].output_channels

            n_features_after_tu_and_concat = 2*nfeatures_from_skip 
        
            self.tu.append(
                    Upsample(
                        n_channels=nfeat_latent, 
                        n_out=nfeatures_from_skip, 
                        scale_factor=up_kernel[u], 
                        mode='bilinear')
                    )
            
            self.conv_blocks_localization.append(BlockDecoder(
                    nfeatures_from_skip, 
                    residual=with_residual, 
                    with_se=with_se))

            self.qz.append(BlockQ(n_features_after_tu_and_concat, nfeatures_from_skip, nfeatures_from_skip))
            self.pz.append(nn.utils.parametrizations.weight_norm(nn.Conv2d(nfeatures_from_skip, nfeatures_from_skip, 1, 1, 0, 1, 1, False), dim=0, name='weight'))

            nfeat_latent = nfeatures_from_skip // 2

        if not self.logger is None:
            self.logger.info(f"Important: reduction from {nfeat_latent} to {nfeat_finalblock}")
        self.final_blocks = {mod:BlockFinalImg(
                nfeat_latent, 
                nfeat_finalblock, 
                num_feat_img, 
                last_act, 
                nb_blocks=nb_finalblocks) \
            for mod in self.modalities}
        
        # register all modules properly
        self.first_conv = nn.ModuleDict(self.first_conv)
        for mod in modalities:
            self.conv_blocks_context[mod] = nn.ModuleList(self.conv_blocks_context[mod])
            self.td[mod] = nn.ModuleList(self.td[mod])
        self.conv_blocks_context = nn.ModuleDict(self.conv_blocks_context)
        self.td = nn.ModuleDict(self.td)
        
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.tu = nn.ModuleList(self.tu)

        self.qz = nn.ModuleList(self.qz)
        self.pz = nn.ModuleList(self.pz)

        self.final_blocks = nn.ModuleDict(self.final_blocks)
        
        self.nb_latent = len(self.conv_blocks_context[modalities[0]])
        
        if self.weightInitializer == 'he':
            self.apply(InitWeights_He(1e-2))

    def create_encodings(self, x):
        skips = {mod:[] for mod in x.keys()}
            
        # Encode each modality independtly
        for mod in x.keys():
            x[mod] = self.first_conv[mod](x[mod])
            for d in range(self.nb_latent - 1):
                x[mod] = self.conv_blocks_context[mod][d](x[mod])
                skips[mod].append(x[mod])
                x[mod] = self.td[mod][d](x[mod])
        
            x[mod] = self.conv_blocks_context[mod][-1](x[mod])
            
        return x, skips
    
    def compute_marginal(self, mu_q_res, log_sigma_q_res, mu_p, inv_sigma_p, temp=1.0):
        inv_sigma_q_res = torch.exp(-log_sigma_q_res)
        
        sigma_q = 1 / (inv_sigma_q_res + inv_sigma_p + 1e-3)
        mu_q = sigma_q * (inv_sigma_q_res * mu_q_res + inv_sigma_p * mu_p)

        # assert not torch.any(torch.isnan(mu_q)), f"{torch.any(torch.isnan(sigma_q))} {torch.any(torch.isnan(inv_sigma_q_res))} {torch.any(torch.isnan(mu_q_res))} {torch.any(torch.isnan(mu_p))}"
        return Normal(mu_q, temp*sigma_q)
    
    def compute_full(self, res_params, prior, temp):
        mu = prior.loc / prior.scale
        inv_sigma = 1 / prior.scale
        
        for mod in res_params.keys():
            mu+= res_params[mod]['loc'] * torch.exp(-res_params[mod]['logscale_res'])
            inv_sigma += torch.exp(-res_params[mod]['logscale_res'])
        mu /= inv_sigma
        sigma = 1 / inv_sigma

        return Normal(mu, temp*sigma)
        

    def encode(self, x):
        """Temp-independent half of forward(): per-modality encoder stack +
        skip connections + residual params (loc, logscale) for q(z_L|x_i).

        None of this depends on `temp` (temp only rescales the Gaussian
        distributions built from these params and the samples drawn from
        them). Call this once per input and reuse the result across multiple
        `decode(..., temp=...)` calls (e.g. the common pattern of sampling
        several temperatures per subject) instead of recomputing the whole
        per-modality encoder stack for every temperature.
        """
        modalities = list(x.keys())  # Corresponds to \pi in paper
        mask = None
        if self.last_act == 'tanh':
            mask = (x[modalities[0]] > -1).float()

        # Create embeddings for injecting info from (x_i)
        x, skips = self.create_encodings(x)

        # Computing residual params for q(z_L|x_i)
        res_params_zL = {}
        for mod in modalities:
            mu_zl_q_res_mod, logvar_zl_q_res_mod = self.bottleneck_down(x[mod]).chunk(2, dim=1)
            mu_zl_q_res_mod = soft_clamp(mu_zl_q_res_mod, self.clamp_value)
            logvar_zl_q_res_mod = soft_clamp(logvar_zl_q_res_mod, self.clamp_value)
            res_params_zL[mod] = {'loc': mu_zl_q_res_mod, 'logscale_res': logvar_zl_q_res_mod}

        return {
            'modalities': modalities,
            'skips': skips,
            'res_params_zL': res_params_zL,
            'mask': mask,
        }

    def decode(self, encoded, temp=1, return_cat=False, return_feat=False, verbose=False,
               target_modality=None, compute_kl=True):
        """Temp-dependent half of forward(): builds q/p distributions level
        by level, samples the hierarchy, and decodes to image space.

        target_modality: if set, only that modality's final block is run
            (skips decoding modalities you're going to throw away downstream).
            Pass None to decode all modalities (original behaviour).
        compute_kl: if False, skips building the per-modality marginal
            distributions and the log_prob evaluations that exist solely to
            populate the `kls` output. Set False whenever the caller discards
            `kls` (e.g. at inference).
        """
        modalities = encoded['modalities']
        skips = encoded['skips']
        mask = encoded['mask']

        # Initialization of distributions and their parameters
        distribs = {f'z{i+1}':dict() for i in range(self.nb_latent)}
        res_params = {f'z{i+1}':dict() for i in range(self.nb_latent)}

        # q(z_L|x_i)
        z_name = 'z{}'.format(self.nb_latent)
        res_params[z_name] = encoded['res_params_zL']
        if compute_kl:
            for mod in modalities:
                rp = res_params[z_name][mod]
                distribs[z_name][mod] = self.compute_marginal(
                    rp['loc'], rp['logscale_res'], torch.tensor(0), torch.tensor(1), temp)  # prior is N(0,I)

        # p(z_L)
        any_mod = modalities[0]
        mu_zl_p = torch.zeros_like(res_params[z_name][any_mod]['loc'])
        sigma_zl_p = torch.ones_like(res_params[z_name][any_mod]['logscale_res'])
        distribs[z_name]['prior'] = Normal(mu_zl_p, sigma_zl_p)

        # Approximate posterior for q(z_L|x_{\pi}) = p(z_L) \prod_{i\in\pi} q(z_L|x_i)
        distribs[z_name]['full'] = self.compute_full(res_params[z_name], distribs[z_name]['prior'], temp)

        # Sampling zL_q from q(z_L|x_{\pi})
        zl_q = distribs[z_name]['full'].rsample()
        if verbose:
            self.logger.info(f"Shape {z_name}: {zl_q.size()}")

        # Computing KLs
        kls = dict()
        if compute_kl:
            for mod in modalities + ['prior']:
                kls[mod] = []
                kl = distribs[z_name]['full'].log_prob(zl_q) - distribs[z_name][mod].log_prob(zl_q)
                kls[mod].append(kl.sum())

        # Creating initial feature volume for z_{L-1}
        zl_q_up = self.bottleneck_up(zl_q)
        z_full = {z_name:zl_q_up}

        for i in range(self.nb_latent - 1): 
            z_name = 'z{}'.format(self.nb_latent-(i+1)) # = z^{l-1}
            
            # Creating feature volume for z_{l-1}
            z_ip1 = z_full['z{}'.format(self.nb_latent-i)]
            x = self.tu[i](z_ip1)
            x = self.conv_blocks_localization[i](x)
            # if verbose:
            #     self.logger.info(f"Shape feature volume for {z_name}: {x.size()}")
            
            # Prior p(z_{l-1}|z_l)
            mu_zi_p, logvar_zi_p = self.pz[i](x).chunk(2, dim=1)
            mu_zi_p = soft_clamp(mu_zi_p, self.clamp_value)
            logvar_zi_p = soft_clamp(logvar_zi_p, self.clamp_value)
            distribs[z_name]['prior'] = Normal(mu_zi_p, torch.exp(logvar_zi_p))
            
            # Computing  q(z_{l-1}|x_i,z_l) 
            for mod in modalities:
                # Merging embedding from z_{l-1} and x_i
                x_q = torch.cat((x, skips[mod][-(i + 1)]), dim=1)
                mu_zi_q_res_mod, logvar_zi_q_res_mod = self.qz[i](x_q).chunk(2, dim=1)
                mu_zi_q_res_mod = soft_clamp(mu_zi_q_res_mod, self.clamp_value)
                logvar_zi_q_res_mod = soft_clamp(logvar_zi_q_res_mod, self.clamp_value)
                res_params[z_name][mod] = {'loc':mu_zi_q_res_mod, 'logscale_res': logvar_zi_q_res_mod}
                if compute_kl:
                    distribs[z_name][mod] = self.compute_marginal(mu_zi_q_res_mod, logvar_zi_q_res_mod, mu_zi_p, torch.exp(-logvar_zi_p), temp) 
            
            # Approximate posterior for q(z_{l-1}|x_{\pi}, z_l) = p(z_{l-1}|z_l) \prod_{i\in\pi} q(z_{l-1}|x_i,z_l)
            distribs[z_name]['full'] = self.compute_full(res_params[z_name], distribs[z_name]['prior'], temp)
        
            # Sampling z_{l-1}
            zi_q = distribs[z_name]['full'].rsample()
            if verbose:
                self.logger.info(f"Shape {z_name}: {zi_q.size()}")

            # Computing KLs
            if compute_kl:
                for mod in modalities + ['prior']:
                    kl = distribs[z_name]['full'].log_prob(zi_q) - distribs[z_name][mod].log_prob(zi_q)
                    kls[mod].append(kl.sum())
            z_full[z_name] = zi_q            

        # Only decode the modality(ies) actually needed downstream
        target_mods = [target_modality] if target_modality is not None else self.modalities

        output_img = {mod:self.final_blocks[mod](z_full['z1']) for mod in target_mods}
        
        if self.last_act=='tanh':
            output_img =  {mod:2*((output_img[mod]+1)/2 * mask) - 1 for mod in target_mods}
            
        if return_cat:
            output_img = torch.cat([output_img[mod] for mod in target_mods], 1)

        if return_feat:
            return  output_img, kls, z_full['z1'] * mask
        else:
            return  output_img, kls        

    def forward(self, x, temp=1, return_cat=False, return_feat=False, verbose=False,
                target_modality=None, compute_kl=True):
        """Unchanged signature/behaviour by default (target_modality=None,
        compute_kl=True reproduce the original forward exactly). Internally
        this now just chains encode() -> decode(); call those directly if
        you want to reuse the encoding across multiple temp values."""
        encoded = self.encode(x)
        return self.decode(encoded, temp=temp, return_cat=return_cat, return_feat=return_feat,
                            verbose=verbose, target_modality=target_modality, compute_kl=compute_kl)
    
       

    def sample(self, batch_size, return_cat=True, temp=0.7):
        
        # Prior distribution for z_L
        mu = torch.zeros((batch_size,2*self.max_features)).cuda()
        sigma = torch.ones((batch_size,2*self.max_features)).cuda()
        p_zl = Normal(mu, temp*sigma)
        
        # Sample from p(z_L)
        zl_p = p_zl.sample()
        zl_p_up = self.bottleneck_up(zl_p)

        z_full = {'z{}'.format(self.nb_latent):zl_p_up}

        for i in range(self.nb_latent - 1):
            z_name = 'z{}'.format(self.nb_latent-(i+1))
            
            # Creating feature volume for z_{l-1}
            z_ip1 = z_full['z{}'.format(self.nb_latent-i)]
            x = self.tu[i](z_ip1)
            x = self.conv_blocks_localization[i](x)

            # Prior p(z_{l-1}|z_l)
            mu_zi_p, logvar_zi_p = self.pz[i](x).chunk(2, dim=1)
            mu_zi_p = soft_clamp(mu_zi_p, self.clamp_value)
            logvar_zi_p = soft_clamp(logvar_zi_p, self.clamp_value)
            var_zi_p = torch.exp(logvar_zi_p)
            p_zi =  Normal(mu_zi_p, temp*var_zi_p)
            
            # Sampling z_{l-1}
            zi_p = p_zi.sample()
            z_full[z_name] = zi_p

        output_img = {mod:self.final_blocks[mod](z_full['z1']) for mod in self.modalities}
        
        if return_cat:
            return torch.cat([output_img[mod] for mod in self.modalities], 1)
        else:
            return output_img


def get_modality_subsets(modalities):
    """All non-empty subsets of `modalities` -- e.g. for 3 input modalities:
    3 singles + 3 pairs + 1 triple = 7 combinations. Each is a structurally
    distinct call into encode()/decode() (different number of dict entries /
    loop trip counts), so each needs its own compiled graph."""
    subsets = []
    for r in range(1, len(modalities) + 1):
        subsets.extend(itertools.combinations(modalities, r))
    return subsets


def compile_mhvae(model, mode='default'):
    """Wrap encode()/decode() with torch.compile.

    encode/decode are compiled separately (rather than the module's
    forward()) so callers can still reuse a single encode() across several
    decode(..., temp=...) calls without re-tracing.

    dynamic=True marks tensor dims (in particular the slice/batch dim, which
    varies per sweep) as symbolic from the start, instead of relying on the
    default "recompile once on a shape change, then go dynamic" heuristic.
    """
    # Each of the 7 modality-subset x (>=2) frame-count combinations produces
    # its own guarded cache entry; the default cache_size_limit (8) is too
    # tight for that, so raise it to avoid silently falling back to eager
    # once the limit is hit.
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)

    model.encode = torch.compile(model.encode, fullgraph=True, mode=mode, dynamic=False)
    model.decode = torch.compile(model.decode, fullgraph=True, mode=mode, dynamic=False)
    return model


@torch.inference_mode()
def warmup_mhvae(model, input_modalities, spatial_shape, warmup_batch_sizes=(24, 40),
                  device='cuda', dtype=torch.float32, target_modality=None, compute_kl=False):
    """Trigger compilation for every (modality-subset x frame-count) pair
    before real inference starts, so the first real sweep of each kind
    doesn't pay a mid-run compile stall.

    input_modalities: the modalities that can appear at the model's input
        (e.g. the 3 MR sequences) -- NOT model.modalities, which also
        includes the synthesis target.
    warmup_batch_sizes: >=2 distinct frame counts. Number of frames in a
        sweep is variable, so this is what lets torch.compile treat that
        dimension as dynamic rather than specializing/recompiling per
        subject on the first real call.
    target_modality: passed through to decode(); defaults to
        model.modalities[0] (the synthesis target), matching real usage.
    """
    if target_modality is None:
        target_modality = model.modalities[0]

    H, W = spatial_shape
    temp_dummy = torch.tensor(1.0, device=device, dtype=dtype)

    for subset in get_modality_subsets(input_modalities):
        for n_slices in warmup_batch_sizes:
            dummy = {
                mod: torch.randn(n_slices, 1, H, W, device=device, dtype=dtype)
                for mod in subset
            }
            # mark_dynamic must be (re)applied to the actual tensors passed
            # in -- it doesn't survive .clone()/new tensor creation, so this
            # has to happen at each call site, including here and in the
            # real inference loop.
            for t in dummy.values():
                torch._dynamo.mark_dynamic(t, 0)

            encoded = model.encode(dummy)
            model.decode(encoded, temp_dummy, return_feat=True, return_cat=True,
                         target_modality=target_modality, compute_kl=compute_kl)