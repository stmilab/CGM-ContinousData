import numpy as np
import pandas as pd
import pdb
from load_24h_cgmacros import generate_24H_CGMacros_dataset


def main():
    time_series_trace_df, meal_event_df, heatmap_fnames = (
        generate_24H_CGMacros_dataset()
    )
    pdb.set_trace()


if __name__ == "__main__":
    main()
