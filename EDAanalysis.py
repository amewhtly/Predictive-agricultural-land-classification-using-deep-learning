import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import rasterio as rio
#import geopandas as gpd
import os
from sklearn.preprocessing import StandardScaler



path = 'C:/Users/milli/OneDrive/Documents/Cranfield University/Thesis/Data/'
pathC = path + 'target/'
pathP = path + 'predictors/'
pathsoil = path + 'soildata/'
outpath = path + 'new/'

soil_path = outpath + "soil_hybrid.tif"
strata_path = pathC + "ALC_predict_hybrid.tif"
raster_paths = [
    #outpath + "soil_hybrid_tag.tif",
    pathP + "AAR.tif",
    pathP + "Aspect.tif",
    pathP + "ASR.tif",
    pathP + "AT0.tif",
    pathP + "ATS.tif",
    pathP + "Channel_Network_Base_Level.tif",
    pathP + "Channel_Network_Distance.tif",
    pathP + "Convergence_Index.tif",
    pathP + "Flow_Path_Length.tif",
    pathP + "LS_Factor.tif",
    pathP + "MRRTF.tif",
    pathP + "MRVBF.tif",
    pathP + "Plan_Curvature.tif",
    pathP + "Profile_Curvature.tif",
    pathP + "Relative_Slope_Position.tif",
    pathP + "Slope.tif",
    pathP + "TCI_Low.tif",
    pathP + "Topographic_Position_Index.tif",
    pathP + "Topographic_Wetness_Index.tif",
    pathP + "Total_Catchment_Area.tif",
    pathP + "Valley_Depth.tif",
    pathP + "DROCK50.tif",
    pathP + "TEXTURE50.tif",
    pathP + "SOILGROUP50.tif",
    pathP + "WETNESS50.tif"
]


categorical_classes = {
    "TEXTURE50": range(1, 22),   # 1–21
    "SOILGROUP50": range(22, 103), # 22–102
    "WETNESS50": range(1, 8)     # 1-7
}

plot_dir = path + "EDAplots"
os.makedirs(plot_dir, exist_ok=True)

stats = {}
combined_data = {}
continuous_rasters = []
categorical_rasters = []

files = raster_paths # soil_hybrid isnt a predictor right? 

for i in files:
    Name = os.path.basename(i)[:-4]  # remove .tif

    with rio.open(os.path.join(path, i)) as src:
        data = src.read(1).astype("float32")
        nodata = src.nodata

        # Clean nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        data[data == 65535] = np.nan
        data[data == -9999] = np.nan
        data[data > 9000] = np.nan

        # Categorical handling
        if Name in categorical_classes:
            valid = categorical_classes[Name]
            data[~np.isin(data, valid)] = np.nan
            categorical_rasters.append(Name)
        else:
            continuous_rasters.append(Name)

        #flat = data.ravel()
        flat = data[~np.isnan(data)]
        if flat.size > 1_000_000:
            flat = np.random.choice(flat, size=1_000_000, replace=False)

        # Save stats
        stats[Name] = {
            "mean": np.nanmean(flat),
            "std": np.nanstd(flat),
            "min": np.nanmin(flat),
            "max": np.nanmax(flat),
            "count": np.sum(~np.isnan(flat))
        }

        if Name not in categorical_classes:
            mean = stats[Name]["mean"]
            std  = stats[Name]["std"]

            # avoid division by zero
            if std > 0:
                combined_data[Name] = flat   # use raw values for analysis
            else:
                norm_flat = flat  # raster is constant

        if Name not in categorical_classes:
            combined_data[Name] = flat


        if Name in categorical_classes:
            print("Category Counts")
            plt.figure(figsize=(8, 6))
            sns.countplot(y=flat[~np.isnan(flat)])
            plt.ylabel(f"{Name} Category")
            plt.xlabel("Count")
            plt.title(f"Category Counts — {Name}")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{Name}_countplot.png"), dpi=150)
            plt.close()

        else:
            print("Histogram")
            plt.figure(figsize=(8,4))
            sns.histplot(flat, bins=30, kde=True)
            plt.xlabel(f"{Name}")
            plt.ylabel("Frequency")
            plt.title(f"Histogram — {Name}")
            plt.savefig(os.path.join(plot_dir, f"{Name}_histogram.png"), dpi=150)
            plt.close()

            print("Boxplot")
            plt.figure(figsize=(8,4))
            sns.boxplot(x=flat)
            plt.xlabel(f"{Name}")
            plt.title(f"Boxplot — {Name}")
            plt.savefig(os.path.join(plot_dir, f"{Name}_boxplot.png"), dpi=150)
            plt.close()

            print("Violin Plot")
            plt.figure(figsize=(8,4))
            sns.violinplot(x=flat)
            plt.xlabel(f"{Name}")
            plt.title(f"Violin — {Name}")
            plt.savefig(os.path.join(plot_dir, f"{Name}_violin.png"), dpi=150)
            plt.close()

# Convert dict of arrays into long-form DataFrame
df_list = []
for name, arr in combined_data.items():
    clean_name = name.replace("_", " ")
    df_list.append(pd.DataFrame({
        "Predictor": clean_name,
        "Value": arr
    }))

df = pd.concat(df_list, ignore_index=True)

# sort predictors by median/variance
# order = df.groupby("Predictor")["Value"].median().sort_values().index
# sns.boxplot(data=df, x="Predictor", y="Value", order=order)

#clip outliers
df["Value"] = df["Value"].clip(-10, 10)

predictors = df["Predictor"].unique().tolist()
chunk_size = 8

for i in range(0, len(predictors), chunk_size):
    chunk = predictors[i:i+chunk_size]
    df_chunk = df[df["Predictor"].isin(chunk)]

    plt.figure(figsize=(2 * len(chunk), 8))
    sns.boxplot(data=df_chunk, x="Predictor", y="Value")
    plt.xticks(rotation=45, ha="right")
    plt.axhline(0, color="black", linewidth=1, linestyle="--")
    #plt.ylim(-10, 10)   # keep scale identical
    plt.title(f"Predictor Distributions — Boxplot (Group {i//chunk_size + 1})")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"predictors_boxplot_group_{i//chunk_size + 1}.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(2 * len(chunk), 8))
    sns.violinplot(data=df_chunk, x="Predictor", y="Value", cut=0)
    plt.xticks(rotation=45, ha="right")
    plt.axhline(0, color="black", linewidth=1, linestyle="--")
    #plt.ylim(-10, 10)   # same scale again
    plt.title(f"Predictor Distributions — Violin (Group {i//chunk_size + 1})")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"predictors_violin_group_{i//chunk_size + 1}.png"), dpi=150)
    plt.close()



###################################################
## correlation matrix ##
# pairwise correlation between continuous rasters

# Clean predictor names once
clean_names = [name.replace("_", " ") for name in continuous_rasters]

# Create empty matrix with cleaned names
corr_matrix = pd.DataFrame(index=clean_names, columns=clean_names, dtype=float)

corr = {}

for i in range(len(continuous_rasters)):
    for j in range(i+1, len(continuous_rasters)):
        A = continuous_rasters[i]
        B = continuous_rasters[j]

        cleanA = A.replace("_", " ")
        cleanB = B.replace("_", " ")

        with rio.open(os.path.join(pathP, A + ".tif")) as srcA:
            a = srcA.read(1).astype("float32")
            nodataA = srcA.nodata
            if nodataA is not None:
                a[a == nodataA] = np.nan
            a[a == 65535] = np.nan
            a[a == -9999] = np.nan
            a[a > 9000] = np.nan
            a = a.ravel()

        with rio.open(os.path.join(pathP, B + ".tif")) as srcB:
            b = srcB.read(1).astype("float32")
            nodataB = srcB.nodata
            if nodataB is not None:
                b[b == nodataB] = np.nan
            b[b == 65535] = np.nan
            b[b == -9999] = np.nan
            b[b > 9000] = np.nan
            b = b.ravel()

        mask = (~np.isnan(a)) & (~np.isnan(b))
        corr_val = np.corrcoef(a[mask], b[mask])[0,1]

        corr_matrix.loc[cleanA, cleanB] = corr_val
        corr_matrix.loc[cleanB, cleanA] = corr_val

# diagonal
for name in clean_names:
    corr_matrix.loc[name, name] = 1


# plot and save
plt.figure(figsize=(12,10))
sns.heatmap(
    corr_matrix.astype(float),
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    annot_kws={"size": 6}
)
plt.title("Correlation matrix - Continuous predictors")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "correlation_matrix.png"))
plt.close()


