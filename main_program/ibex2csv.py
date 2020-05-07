
from detection import detect_bex as detect_bex
from classification import filter_bex as filter_ibex

def main():
    filename_properties = filter_ibex.main()
    detect_bex.main(filename_properties)

if __name__ == "__main__":
    main()