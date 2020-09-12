import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv("medical_examination.csv")

# Add 'overweight' column
df['overweight'] = np.where(df["weight"]/(df["height"]*0.01)**2 > 25, 1, 0)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholestorol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df["cholesterol"] = np.where(df["cholesterol"] == 1, 0, 1)
df["gluc"] = np.where(df["gluc"] == 1, 0, 1)

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = df.melt(id_vars=["cardio"], value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"])

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the collumns for the catplot to work correctly.
    df_cat = df_cat.groupby(["variable", "cardio", "value"]).agg(total=("value", "count")).reset_index()

    # Draw the catplot with 'sns.catplot()'
    plt.figure()
    fig = sns.catplot(data=df_cat, x="variable", y="total", hue="value", col="cardio", kind="bar").fig

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    mask_pressure = df["ap_lo"] <= df["ap_hi"]
    mask_height = (df["height"] >= df["height"].quantile(0.025)) & (df["height"] <= df["height"].quantile(0.975))
    mask_weight = (df["weight"] >= df["weight"].quantile(0.025)) & (df["weight"] <= df["weight"].quantile(0.975))
    df_heat = df.loc[mask_pressure & mask_height & mask_weight]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(data=corr, annot=True, fmt=".1f", cmap="Reds", mask=mask, ax=ax)

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
