import argparse
from utilities.sweep_generator import SyntheticSweepConfig, SynthesisConfig, SyntheticSweepGenerator

def parse_args():
    parser = argparse.ArgumentParser(
        prog="GenerateSyntheticSweeps",
        description="Generate synthetic ultrasound sweeps from MRI data"
    )
    parser.add_argument("--case", type=str, required=True, help="Case identifier, e.g. Case027")
    parser.add_argument("--annotator", type=str, default="n1", help="Annotator")
    parser.add_argument("--K", type=int, default=1, help="Number of sweeps per modality subset")
    parser.add_argument("--fold", type=int, default=0, help="Synthesizer fold to use for synthesis")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()

    config = SyntheticSweepConfig()
    generator = SyntheticSweepGenerator(config)

    generator.generate_case_and_synthesize(
        case=args.case,
        annotator=args.annotator,
        num_sweeps_per_subset=args.K,
        seed=args.seed,
        synthesis_cfg=SynthesisConfig(
            model_dir=f"models/synthesis/mmhvae_f{args.fold}",
            case=args.case
        )
    )

if __name__ == "__main__":
    main()