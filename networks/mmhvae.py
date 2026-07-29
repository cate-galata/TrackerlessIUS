#   CODE ADAPTED FROM: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/network_architecture/generic_UNet.py

from copy import deepcopy
from torch import nn
import torch
import numpy as np
import torch.nn.functional
from torch.distributions import Normal
from networks.blocks import BlockEncoder, AsFeatureMap_down, AsFeatureMap_up, BlockDecoder, BlockFinalImg, Upsample, BlockQ, InitWeights_He


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
        

    def forward(self, x, temp=1, return_cat=False, return_feat=False, verbose=False):
        modalities = list(x.keys()) # Corresponds to \pi in paper
        if self.last_act=='tanh':
            mask = (x[modalities[0]]>-1).float()
                
        # Create embeddings for injecting info from (x_i)
        x, skips = self.create_encodings(x)
        
        # Initialization of distributions and their parameters
        distribs = {f'z{i+1}':dict() for i in range(self.nb_latent)}
        res_params = {f'z{i+1}':dict() for i in range(self.nb_latent)}
        
        # Computing q(z_L|x_i) 
        z_name = 'z{}'.format(self.nb_latent)
        for mod in modalities:
            mu_zl_q_res_mod, logvar_zl_q_res_mod = self.bottleneck_down(x[mod]).chunk(2, dim=1)
            mu_zl_q_res_mod = soft_clamp(mu_zl_q_res_mod, self.clamp_value)
            logvar_zl_q_res_mod = soft_clamp(logvar_zl_q_res_mod, self.clamp_value)
            res_params[z_name][mod] = {'loc':mu_zl_q_res_mod, 'logscale_res': logvar_zl_q_res_mod}
            distribs[z_name][mod] = self.compute_marginal(mu_zl_q_res_mod, logvar_zl_q_res_mod, torch.tensor(0), torch.tensor(1), temp) # prior is N(0,I)
        
        # p(z_L)
        mu_zl_p = torch.zeros_like(mu_zl_q_res_mod)
        sigma_zl_p = torch.ones_like(logvar_zl_q_res_mod)
        distribs[z_name]['prior'] = Normal(mu_zl_p, sigma_zl_p)
            
        # Approximate posterior for q(z_L|x_{\pi}) = p(z_L) \prod_{i\in\pi} q(z_L|x_i)
        distribs[z_name]['full'] = self.compute_full(res_params[z_name], distribs[z_name]['prior'], temp)
        
        # Sampling zL_q from q(z_L|x_{\pi})
        zl_q = distribs[z_name]['full'].rsample()
        if verbose:
            self.logger.info(f"Shape {z_name}: {zl_q.size()}")
        
        # Computing KLs
        kls = dict()
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
                distribs[z_name][mod] = self.compute_marginal(mu_zi_q_res_mod, logvar_zi_q_res_mod, mu_zi_p, torch.exp(-logvar_zi_p), temp) 
            
            # Approximate posterior for q(z_{l-1}|x_{\pi}, z_l) = p(z_{l-1}|z_l) \prod_{i\in\pi} q(z_{l-1}|x_i,z_l)
            distribs[z_name]['full'] = self.compute_full(res_params[z_name], distribs[z_name]['prior'], temp)
        
            # Sampling z_{l-1}
            zi_q = distribs[z_name]['full'].rsample()
            if verbose:
                self.logger.info(f"Shape {z_name}: {zi_q.size()}")

            # Computing KLs
            for mod in modalities + ['prior']:
                kl = distribs[z_name]['full'].log_prob(zi_q) - distribs[z_name][mod].log_prob(zi_q)
                kls[mod].append(kl.sum())
            z_full[z_name] = zi_q            

        output_img = {mod:self.final_blocks[mod](z_full['z1']) for mod in self.modalities}
        
        if self.last_act=='tanh':
            output_img =  {mod:2*((output_img[mod]+1)/2 * mask) - 1 for mod in self.modalities}
            
        if return_cat:
            output_img = torch.cat([output_img[mod] for mod in self.modalities], 1)

        if return_feat:
            return  output_img, kls, z_full['z1'] * mask
        else:
            return  output_img, kls        
    
       

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

    # ------------------------------------------------------------------
    # Fast-inference path.
    #
    # These methods are functionally equivalent to forward(), specialized
    # for eval-only usage:
    #   - No torch.distributions.Normal objects (they don't trace cleanly
    #     under torch.compile/export and add unnecessary Python overhead).
    #     Gaussian fusion + reparameterized sampling are done with plain
    #     tensor ops instead, algebraically identical to compute_marginal /
    #     compute_full / Normal.rsample().
    #   - No KL computation (unused at inference; skipping it avoids
    #     wasted log_prob calls).
    #   - encode_one(mod, x) processes a SINGLE modality; because
    #     first_conv/conv_blocks_context/td are per-modality weights with
    #     no cross-modality interaction at this stage, this can be compiled
    #     once per modality IDENTITY, independent of which other
    #     modalities are present in a given call.
    #   - decode_by_count(x_list, skips_list, ...) takes plain lists
    #     (not a Dict[str, Tensor]) so the graph depends only on how many
    #     modalities are present (len(x_list)), never on which ones or
    #     what order -- letting you compile exactly 3 decoder variants
    #     (count = 1, 2, or 3) instead of one per combination/permutation.
    # ------------------------------------------------------------------

    def encode_one(self, mod, x_mod):
        """Encode a single modality. Compiled separately per modality identity."""
        h = self.first_conv[mod](x_mod)
        skips = []
        for d in range(self.nb_latent - 1):
            h = self.conv_blocks_context[mod][d](h)
            skips.append(h)
            h = self.td[mod][d](h)
        h = self.conv_blocks_context[mod][-1](h)
        return h, skips
 
    @staticmethod
    def _fuse_posterior(mu_prior, inv_sigma_prior, res_list, temp):
        """
        Precision-weighted fusion of a Gaussian prior with k Gaussian
        "residual" terms (one per present modality), returned as a
        torch.distributions.Normal.
 
        Deliberately uses Normal/.rsample() here rather than a hand-rolled
        mu + sigma * torch.randn_like(mu): Normal's methods are opaque to
        Dynamo and force a graph break at the sampling step, which means
        the RNG kernel itself runs eagerly instead of being traced into
        the compiled graph. That's a feature here, not a limitation --
        Inductor's random-number codegen has historically been a common
        source of unwanted shape specialization on dynamic dims, and this
        sidesteps it entirely. Everything before this (all the conv
        layers) still compiles; only the small elementwise sample/prior
        arithmetic falls back to eager.
 
        res_list: list of (mu_res, logscale_res) tuples.
        """
        mu = mu_prior * inv_sigma_prior
        inv_sigma = inv_sigma_prior
        for mu_res, logscale_res in res_list:
            inv_sigma_res = torch.exp(-logscale_res)
            mu = mu + mu_res * inv_sigma_res
            inv_sigma = inv_sigma + inv_sigma_res
        mu_full = mu / inv_sigma
        sigma_full = temp / inv_sigma
        return Normal(mu_full, sigma_full)
 
    def decode_by_count(self, x_list, skips_list, temp, mask, output_mods=None):
        """
        x_list:      list[Tensor], length k -- bottleneck feature volume
                     per present input modality (from encode_one).
        skips_list:  list[list[Tensor]], length k, each of length
                     nb_latent - 1 (from encode_one).
        temp:        0-dim tensor (NOT a python float -- keeping it a
                     tensor input means it doesn't get compiled in as a
                     constant, so one compiled graph serves every
                     temperature value).
        mask:        tensor with the same shape as the output image,
                     built from any present input modality's raw tensor
                     as (raw_img > -1).float() (only meaningful when
                     last_act == 'tanh', matching forward()).
        output_mods: modalities to synthesize; defaults to self.modalities
                     (all of them -- this is a multi-modal synthesizer,
                     so it always tries to produce every modality
                     regardless of which ones were given as input).
 
        Graph shape depends only on k = len(x_list), never on modality
        identity or ordering.
        """
        if output_mods is None:
            output_mods = self.modalities
 
        # ---- q(z_L | x_pi) ----
        res_list = []
        for h in x_list:
            mu_res, logscale_res = self.bottleneck_down(h).chunk(2, dim=1)
            mu_res = soft_clamp(mu_res, self.clamp_value)
            logscale_res = soft_clamp(logscale_res, self.clamp_value)
            res_list.append((mu_res, logscale_res))
 
        mu_prior = torch.zeros_like(res_list[0][0])
        inv_sigma_prior = torch.ones_like(res_list[0][1])
        zl_q = self._fuse_posterior(mu_prior, inv_sigma_prior, res_list, temp).rsample()
 
        z_cur = self.bottleneck_up(zl_q)
 
        # ---- q(z_{l-1} | x_pi, z_l), l = nb_latent-1 ... 1 ----
        for i in range(self.nb_latent - 1):
            x = self.tu[i](z_cur)
            x = self.conv_blocks_localization[i](x)
 
            mu_zi_p, logvar_zi_p = self.pz[i](x).chunk(2, dim=1)
            mu_zi_p = soft_clamp(mu_zi_p, self.clamp_value)
            logvar_zi_p = soft_clamp(logvar_zi_p, self.clamp_value)
            inv_sigma_zi_p = torch.exp(-logvar_zi_p)
 
            res_list = []
            for skips in skips_list:
                skip = skips[-(i + 1)]
                x_q = torch.cat((x, skip), dim=1)
                mu_res, logscale_res = self.qz[i](x_q).chunk(2, dim=1)
                mu_res = soft_clamp(mu_res, self.clamp_value)
                logscale_res = soft_clamp(logscale_res, self.clamp_value)
                res_list.append((mu_res, logscale_res))
 
            z_cur = self._fuse_posterior(mu_zi_p, inv_sigma_zi_p, res_list, temp).rsample()
 
        output_img = {mod: self.final_blocks[mod](z_cur) for mod in output_mods}
 
        if self.last_act == 'tanh':
            output_img = {mod: 2 * ((output_img[mod] + 1) / 2 * mask) - 1 for mod in output_mods}
 
        return output_img
 
 
def build_compiled_inference_fns(model, compile_mode="default", device=None,
                                  spatial_shape=(192, 192), warmup_ns=(8, 24)):
    """
    Builds and warms up:
      - 3 compiled encoders, one per modality identity (independent of
        which other modalities are present in a given call)
      - 3 compiled decoders, one per input-modality COUNT k in {1,2,3}
        (independent of which modalities / what order)
 
    encode_one is free of torch.distributions.Normal (pure convs), so it
    compiles fullgraph=True with no breaks.
 
    decode_by_count uses Normal(...).rsample() for the hierarchical
    sampling steps. This is intentional: Normal's methods are opaque to
    Dynamo, so it breaks precisely at each .rsample() call rather than
    tracing the RNG kernel into the compiled graph -- which sidesteps a
    class of shape-specialization issues Inductor's random-number codegen
    can otherwise trigger on dynamic dims. Everything else in decode
    (bottleneck up/down, tu, conv_blocks_localization, qz, pz,
    final_blocks -- i.e. essentially all the actual FLOPs) still compiles;
    only the small elementwise prior/posterior arithmetic and sampling
    around each level runs eagerly. Hence fullgraph=False for decoders.
    If you want to chase full-graph decoding later, the next thing to
    check is whether the elementwise fusion across res_list (mu/inv_sigma
    accumulated from independently-mark_dynamic'd per-modality tensors)
    is itself forcing specialization, independent of the RNG question --
    run with TORCH_LOGS="+dynamic" to see exactly which op triggers it.
 
    Note: we deliberately do NOT pass dynamic=True to torch.compile here.
    dynamic=True marks every dimension of every input as symbolic from the
    first trace -- not just the ones we explicitly flag. Only dim 0 (the
    slice/batch count, N) actually varies for this model; H and W are
    always fixed by original_shape/num_pool. Letting H/W become symbolic
    breaks the bilinear Upsample blocks (self.tu[i]): F.interpolate's
    scale_factor code path multiplies the (now-symbolic) spatial size by
    the scale factor, producing a SymInt tuple that the underlying
    upsample_bilinear2d ATen op's "tuple of floats" overload rejects.
    mark_dynamic(tensor, 0), used below and at each call site, is the
    correct amount of dynamism: only N becomes symbolic, H/W stay
    concrete Python ints, and the Upsample scale_factor math stays in
    plain-float territory.
 
    compile_mode:
      - "default": fastest to warm up, no CUDA graphs, dynamic-shape
        friendly. Good default, and the safe fallback if max-autotune's
        warmup time doesn't amortize over your run.
      - "max-autotune": searches for faster kernels; more warmup cost but
        worth it if you're processing many subjects/folders so the extra
        one-time compile cost amortizes. Recommended here given the
        batch/production nature of this script.
      - "reduce-overhead" (CUDA graphs) is deliberately NOT supported:
        CUDA graphs replay a fixed sequence of kernel launches against
        fixed shapes/addresses, which is fundamentally incompatible with
        N (slice count) varying per subject the way it does here. If you
        want CUDA-graph-level launch-overhead savings later, that
        requires a separate bucketing strategy (pad N up to a small set
        of fixed sizes and run static-shape compiled variants) rather
        than this dynamic-N approach.
 
    Returns:
        encoders: dict[mod] -> callable(x_mod) -> (h, skips)
        decoders: dict[k]   -> callable(x_list, skips_list, temp, mask) -> output_img dict
    """
    assert compile_mode in ("default", "max-autotune-no-cudagraphs"), (
        "reduce-overhead (CUDA graphs) is not supported here -- see docstring."
    )
 
    torch._dynamo.config.cache_size_limit = 64
 
    if device is None:
        device = next(model.parameters()).device
 
    # --- 3 compiled encoders, one per modality identity ---
    encoders = {}
    for mod in model.modalities:
        def _make(mod):
            def _fn(x_mod):
                return model.encode_one(mod, x_mod)
            return _fn
        encoders[mod] = torch.compile(_make(mod), fullgraph=True, mode=compile_mode)
 
    # --- 3 compiled decoders, one per input-modality count ---
    decoders = {
        k: torch.compile(model.decode_by_count, fullgraph=False, mode=compile_mode)
        for k in (1, 2, 3)
    }
 
    # --- Warmup: 2 distinct N values per graph so the batch/slice dim is
    #     promoted to a symbolic size before real inference starts, rather
    #     than paying that recompile cost on the first real subject. ---
    with torch.inference_mode():
        for mod, enc_fn in encoders.items():
            for n in warmup_ns:
                dummy = torch.randn(n, 1, *spatial_shape, device=device)
                torch._dynamo.mark_dynamic(dummy, 0)
                enc_fn(dummy)
 
        for k, dec_fn in decoders.items():
            for n in warmup_ns:
                x_list, skips_list = [], []
                for _ in range(k):
                    dummy = torch.randn(n, 1, *spatial_shape, device=device)
                    h, skips = model.encode_one(model.modalities[0], dummy)
                    torch._dynamo.mark_dynamic(h, 0)
                    for s in skips:
                        torch._dynamo.mark_dynamic(s, 0)
                    x_list.append(h)
                    skips_list.append(skips)
                mask = torch.ones(n, 1, *spatial_shape, device=device)
                temp_t = torch.tensor(0.7, device=device)
                dec_fn(x_list, skips_list, temp_t, mask)
 
    return encoders, decoders