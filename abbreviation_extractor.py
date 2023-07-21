import sys
sys.path.append("/data/yuzhao/code/PubMedSearcher/Ab3P/Ab3P_v1_5/Library/")
import ab3p

class AbbreviationExtractor(object):
    def __init__(self) -> None:
        self.AB3P = ab3p.Ab3P()

    def get_abbreviations(self, text):
        self.AB3P.get_abbrs(text, [])

def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        '--text', 
        type=str, 
        default="protein expression of brain-derived neurotrophic factor (BDNF) and cAMP-response element-binding protein (CREB) phosphorylation."
    )
    args = parser.parse_args()

    AEor = AbbreviationExtractor()
    AEor.get_abbreviations(args.text)

if __name__ == "__main__":
    main()

