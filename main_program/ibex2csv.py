
from detection import detect_bex as detect_bex
from classification import filter_bex as filter_ibex

def main():
    filter_ibex.main()
    detect_bex.main()

if __name__ == "__main__":
    main()