import numpy as np
import pandas as pd
from scipy import stats


discount_yes = Fact_Sales.loc[Fact_Sales["Discount"] > 0].assign(
    Discount_type="Discounted"
)
discount_no = Fact_Sales.loc[Fact_Sales["Discount"] == 0].assign(
    Discount_type="No Discount"
)

combined_df = pd.concat([discount_no, discount_yes], ignore_index=True)


quantity_counts_df = (
    combined_df.groupby(["Quantity", "Discount_type"], as_index=False)
    .size()
    .rename(columns={"size": "Order_Count"})
)


quantity_probability_df = (
    quantity_counts_df.assign(
        Group_Total=lambda d: d.groupby("Discount_type")["Order_Count"].transform("sum")
    )
    .assign(Probability=lambda d: d["Order_Count"] / d["Group_Total"])
    .drop(columns="Group_Total")
)

group_size_df = (
    combined_df.groupby("Discount_type", as_index=False)
    .agg(Order_Count=("Quantity", "size"))
    .assign(Group_Percent=lambda d: d["Order_Count"] / d["Order_Count"].sum())
)


def yeo_summary(series: pd.Series, label: str):
    transformed, lam = stats.yeojohnson(series)
    sample = (
        transformed
        if transformed.shape[0] <= 5000
        else np.random.choice(transformed, 5000, replace=False)
    )
    stat, p_val = stats.shapiro(sample)
    osm, osr = stats.probplot(transformed, dist="norm", fit=False)
    qq = pd.DataFrame({"Theoretical": osm, "Ordered": osr, "Discount_type": label})
    summary = pd.DataFrame(
        {
            "Discount_type": [label],
            "YeoJohnson_lambda": [lam],
            "Shapiro_statistic": [stat],
            "Shapiro_p_value": [p_val],
        }
    )
    return summary, qq


yj_discounted_df, qq_discounted_df = yeo_summary(discount_yes["Quantity"], "Discounted")
yj_nodiscount_df, qq_nodiscount_df = yeo_summary(discount_no["Quantity"], "No Discount")

yeo_johnson_summary_df = pd.concat(
    [yj_discounted_df, yj_nodiscount_df], ignore_index=True
)
qq_plot_df = pd.concat([qq_discounted_df, qq_nodiscount_df], ignore_index=True)


mw_stat, mw_p_value = stats.mannwhitneyu(
    discount_yes["Quantity"], discount_no["Quantity"], alternative="two-sided"
)
mannwhitney_summary_df = pd.DataFrame(
    {
        "Test": ["Mann-Whitney U"],
        "Statistic": [mw_stat],
        "p_value": [mw_p_value],
        "alpha(0.05)_Reject_H0": [mw_p_value < 0.05],
    }
)

group_moments_df = combined_df.groupby("Discount_type", as_index=False)["Quantity"].agg(
    Mean="mean", Median="median", StdDev="std"
)
