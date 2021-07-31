import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from mlxtend.plotting import category_scatter
from pyecharts.charts import Map,Geo
from pyecharts import options as opts
from pyecharts.globals import ThemeType
import os
import qrcode

# make directory plots
if not os.path.isdir("Plots"):  # if the directory Plots doesn’t exist:
  os.makedirs("./Plots") # make it
# data
data_raw = pd.read_csv("https://covid.ourworldindata.org/data/owid-covid-data.csv", sep = ",")

# scaling positive rate, calculating case fatality ratio corrected for positive rate
def scale(data, colname):
  mean = data.loc[:, colname].mean()
  sd = data.loc[:, colname].std()
  rescale = (data.loc[:, colname] - mean) / sd
  return rescale

data_raw["positive_rate_scaled"] = (scale(data_raw, "positive_rate") + 1)
data_raw["case_fatality_ratio"] = data_raw["total_deaths_per_million"] / (data_raw["total_cases_per_million"] * data_raw["positive_rate_scaled"])
data_raw["case_fatality_ratio_uncorrected"] = data_raw["total_deaths_per_million"] / (data_raw["total_cases_per_million"])
data_raw["CFR_log"] = np.log(data_raw["case_fatality_ratio"])

# new dataframe including only relevant variables and dropping NAs
data = data_raw.iloc[:, [0, 1, 2, 3, 44, 42, 43, 25, 27, 28, 29, 30, 31, 33, 34, 35, 36, 38, 39, 40]].dropna().copy()

# descriptives
def descriptives(dataset, selected_continent, variable):
  if selected_continent is "World":
    descriptives = {
      "Count": dataset.loc[:, variable].describe()[0],
      "Mean": dataset.loc[:, variable].mean(),
      "SD": dataset.loc[:, variable].std(),
      "Min": dataset.loc[:, variable].min(),
      "Max": dataset.loc[:, variable].max(),
      "Q1": dataset.loc[:, variable].describe()[4],
      "Q3": dataset.loc[:, variable].describe()[6],
      "IQR": stats.iqr(dataset.loc[:, variable])
      }
    return (descriptives)
  else:
        selection = dataset.loc[:, "continent"] == selected_continent
        descriptives = {
          "Count": dataset.loc[selection, variable].describe()[0],
          "Mean": dataset.loc[selection, variable].mean(),
          "SD": dataset.loc[selection, variable].std(),
          "Min": dataset.loc[selection, variable].min(),
          "Max": dataset.loc[selection, variable].max(),
          "Q1": dataset.loc[selection, variable].describe()[4],
          "Q3": dataset.loc[selection, variable].describe()[6],
          "IQR": stats.iqr(dataset.loc[selection, variable])
          }
  return (descriptives)

continents = data.loc[:, "continent"].unique()
descriptives_table = {}
for i in continents:
    descriptives_table[i] = descriptives(data, i, "case_fatality_ratio")
    
descriptives_table = pd.DataFrame.from_dict(descriptives_table)
descriptives_table_world = pd.DataFrame.from_dict(descriptives(data, "World", "case_fatality_ratio"), orient = "index")
descriptives_table_final = descriptives_table.join(descriptives_table_world)
descriptives_table_final.columns.values[-1] = "World"
descriptives_table_final

pd.set_option('display.max_columns', None) # print full table
print(descriptives_table_final)

# histogram CFR
def plot_hist(dataset, var, color, xlabel, title, savelocation):
  fig, ax = plt.subplots(1,1)
  n, bins, patches = ax.hist(dataset[var], bins = 100, density = 100, color = color)
  ax.set_xlabel(xlabel)
  ax.set_ylabel("Count")
  ax.set_title(title)
  plt.show()
  plt.savefig(savelocation, dpi = 1080)
  plt.close()
  return

plot_hist(data, "case_fatality_ratio", "green", "Case Fatality Ratio", "Histogram of the Case Fatality Ratio", "./Plots/hist1.png")
plot_hist(data, "case_fatality_ratio_uncorrected", "blue", "Uncorrected Case Fatality Ratio", "Histogram of the Uncorrected Case Fatality Ratio", "./Plots/hist2.png")
plot_hist(data, "CFR_log", "red", "Case Fatality Ratio Log", "Histogram of the Logarithmic Transformed Case Fatality Ratio", "./Plots/hist3.png")

# correlation matrix
# define plot
fig, ax = plt.subplots(1,1)
# correlation and labels
corr = data.iloc[:, 4:len(data.columns)].corr()
labels = ("CFR Log", "CFR", "CFR Uncorrected", "Stringency index",
  "Population density", "Median age", "Aged 65 older",
  "Aged 70 older", "GDP per capita", "CVD death rate",
  "Diabetes prevalence", "Smokers F", "Smokers M",
  "Hospital beds", "Life expectancy", "HDI")
# plot specifications
ax = sns.heatmap(
    corr, 
    vmin = -1, vmax = 1, center = 0,
    cmap = "Spectral",
    square = True)
ax.set_xticklabels(
    labels,
    size = 8,
    rotation = 25,
    horizontalalignment = "right")
ax.set_yticklabels(
    labels,
    size = 8,
    rotation = 25,
    verticalalignment = "top")
plt.title("Correlation Covid-related Variables", fontsize = 18)
plt.show()
plt.savefig(fname = "./Plots/corrmatrix.png", dpi = 1080)
plt.close()

# gee models
fam = sm.families.Gaussian()
ind = sm.cov_struct.Exchangeable()
mod_gee = smf.gee("CFR_log ~ stringency_index + population_density + median_age + aged_65_older + aged_70_older + gdp_per_capita + cardiovasc_death_rate + diabetes_prevalence + female_smokers + male_smokers + hospital_beds_per_thousand + life_expectancy + human_development_index", "location", data = data, cov_struct = ind, family = fam) 
result_gee = mod_gee.fit()
print(result_gee.summary())

mod_gee2 = smf.gee("case_fatality_ratio ~ stringency_index + population_density + median_age + aged_65_older + aged_70_older + gdp_per_capita + cardiovasc_death_rate + diabetes_prevalence + female_smokers + male_smokers + hospital_beds_per_thousand + life_expectancy + human_development_index", "location", data = data, cov_struct = ind, family = fam) 
result_gee2 = mod_gee2.fit()
print(result_gee2.summary())

mod_gee3 = smf.gee("case_fatality_ratio_uncorrected ~ stringency_index + population_density + median_age + aged_65_older + aged_70_older + gdp_per_capita + cardiovasc_death_rate + diabetes_prevalence + female_smokers + male_smokers + hospital_beds_per_thousand + life_expectancy + human_development_index", "location", data = data, cov_struct = ind, family = fam) 
result_gee3 = mod_gee3.fit()
print(result_gee3.summary())


# scatterplots
fig = category_scatter(x = "date", y = "case_fatality_ratio", label_col = "continent", data = data, legend_loc = "best")
fig.suptitle("CFR over time for each continent")
plt.xlabel("Time")
plt.ylabel("CFR")
plt.show()
plt.savefig(fname = "./Plots/scatter1.png", dpi = 1080)
plt.close()

fig = category_scatter(x = "date", y = "case_fatality_ratio_uncorrected", label_col = "continent", data = data, legend_loc = "best")
fig.suptitle("Uncorrected CFR over time for each continent")
plt.xlabel("Time")
plt.ylabel("Uncorrected CFR")
plt.show()
plt.savefig(fname = "./Plots/scatter2.png", dpi = 1080)
plt.close() 

fig = category_scatter(x = "case_fatality_ratio", y = "case_fatality_ratio_uncorrected", label_col = "continent", data = data, legend_loc = "best")
fig.suptitle("Scatterplot CFR corrected vs uncorrected")
plt.xlabel("Corrected CFR")
plt.ylabel("Uncorrected CFR")
plt.show()
plt.savefig(fname = "./Plots/scatter3.png", dpi = 1080)
plt.close()

selection = data.loc[:, "continent"] == "Europe"
fig = category_scatter(x = "case_fatality_ratio", y = "case_fatality_ratio_uncorrected", label_col = "location", data = data.loc[selection,:], legend_loc = False)
fig.suptitle("Scatterplot CFR corrected vs uncorrected for Europe")
plt.xlabel("Corrected CFR")
plt.ylabel("Uncorrected CFR")
plt.show()
plt.savefig(fname = "./Plots/scatter4.png", dpi = 1080)
plt.close()



# map
#change date from object data type to datetime data type
data_map = data.loc[:, ["iso_code", "continent", "location", "date", "case_fatality_ratio", "case_fatality_ratio_uncorrected"]]
data_map["date"] = pd.to_datetime(data_map["date"])
data_map = data_map.loc[data_map.groupby("location").date.idxmax()].sort_values(by = ["date"], ascending = False)
data_map
# lists for projection on map
country = list(data_map["location"])
case_fatality_ratio = list(data_map["case_fatality_ratio"])

# 
data_map.loc[:, "case_fatality_ratio"].describe() # for choosing thresholds visualmap options
max_visual = np.max(data_map.loc[:, "case_fatality_ratio"]) # highest CFR

list1 = [[country[i], case_fatality_ratio[i]] for i in range(len(country))] # prepare data for visualization
map_CFR = Map(init_opts = opts.InitOpts(width = "1000px", height = "460px", theme = ThemeType.ROMANTIC)) # create the map and set the size of the map
map_CFR.add("Case Fatality Ratio per Country", list1, maptype = "world", is_map_symbol_show = False) # add world map
map_CFR.set_series_opts(label_opts = opts.LabelOpts(is_show = False))
map_CFR.set_global_opts(
  visualmap_opts = 
    opts.VisualMapOpts(
      max_ = max_visual, is_piecewise = True, pieces = [
        {"min": data_map.loc[:, "case_fatality_ratio"].describe()[3], "max": (data_map.loc[:, "case_fatality_ratio"].describe()[4] - 1e-10), "label": "Below first quartile"},
        {"min": data_map.loc[:, "case_fatality_ratio"].describe()[4], "max": (data_map.loc[:, "case_fatality_ratio"].describe()[5] - 1e-10), "label": "Below median"}, 
        {"min": data_map.loc[:, "case_fatality_ratio"].describe()[5], "max": (data_map.loc[:, "case_fatality_ratio"].describe()[6] - 1e-10), "label": "Above median"}, 
        {"min": data_map.loc[:, "case_fatality_ratio"].describe()[6], "max": data_map.loc[:, "case_fatality_ratio"].describe()[7], "label": "Above third quartile"}]), 
    title_opts = opts.TitleOpts(
      title = "Worldwide Covid-19 Case Fatality Rate Corrected for Positive Rate",
      subtitle = "Most recently available data used from doi: 10.1038/s41597-020-00688-8",
      pos_left = "center",
      padding = 0,
      item_gap = 2,
      title_textstyle_opts = opts.TextStyleOpts(
        color = "firebrick", 
        font_weight = "bold", 
        font_family = "Times New Roman", 
        font_size = 30),
      subtitle_textstyle_opts = opts.TextStyleOpts(
        color = "darkblue", 
        font_weight = "bold", 
        font_family = "Times New Roman", 
        font_size = 13)),
    legend_opts = opts.LegendOpts(is_show = False))
map_CFR.render(path = "Plots/map.html")

#make qr code plot map
img = qrcode.make("https://github.com/TimFinke/AssignmentPython/blob/main/map.html")
img.save("Plots/mapqr.png")
