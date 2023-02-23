import calendar
from datetime import timedelta
from colorama import Fore
import joblib
from matplotlib import dates
from matplotlib.widgets import Slider
import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
#pip install scikit-learn

def clean_data(data):
    return data.dropna()

def exploratory_data_analysis(data):
    print(data.describe())
    print(data.dtypes)
    print(data.shape)
    print(data.isnull().sum())

def check_wind_trends(data):
    data = data[data.index.month == 1]
    plt.plot(data.index, data['Wind Speed (m/s)'], label = data.index, linewidth = 1, color = 'red')
    plt.show()


def optimize_dateformat(data):
    data["Date"] = pd.to_datetime(data['Date/Time'], format='%d %m %Y %H:%M')
    data.drop('Date/Time', axis = 1, inplace = True)
    data.set_index("Date", inplace = True)
    return data

def set_dark_bg():
    mpl.style.use('dark_background')
    sns.set_palette('bright')

def clearPlts():
    plt.clf()
    plt.close("all")

def most_productive_periods(data, period = 'month'):
    if period == 'hour':
        avg_Kw = data.groupby(data.index.hour)['LV ActivePower (kW)'].mean()
        avg_Wind_speed = data.groupby(data.index.hour)['Wind Speed (m/s)'].mean()
        x_label = 'Hour of the day'
        # calculate ratio of LV ActivePower to Wind Speed
        ratio = data.groupby(data.index.hour)['LV ActivePower (kW)'].mean() / data.groupby(data.index.hour)['Wind Speed (m/s)'].mean()
    else:
        avg_Kw = data.groupby(data.index.month)['LV ActivePower (kW)'].mean()
        avg_Wind_speed = data.groupby(data.index.month)['Wind Speed (m/s)'].mean()
        x_label = 'Months of the year'
        ratio = data.groupby(data.index.month)['LV ActivePower (kW)'].mean() / data.groupby(data.index.month)['Wind Speed (m/s)'].mean()

    fig, ax1 = plt.subplots()

    #set_dark_bg()
    color = 'yellow'
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Average LV ActivePower (kW)', color = color)
    ax1.plot(avg_Kw.index, avg_Kw.values, color = color)
    ax1.tick_params(axis='y', labelcolor = color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'lightblue'
    ax2.set_ylabel('Average Wind Speed (m/s)', color = color)  # we already handled the x-label with ax1
    ax2.plot(avg_Wind_speed.index, avg_Wind_speed.values, color = color)
    ax2.tick_params(axis='y', labelcolor = color)
        # Add horizontal lines at the average wind speed and average LV ActivePower
    ax2.axhline(y=avg_Wind_speed.values.mean(), color = 'blue', linestyle = '--')
    ax1.axhline(y=avg_Kw.values.mean(), color = 'orange', linestyle = '--')
    
    # find the highest ratio
    best = ratio.idxmax()
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(ratio)
    print(best)
    print(avg_Kw)
    print(avg_Wind_speed)
     # Add a vertical line at the highest product value
    ax1.axvline(x=best, color='red', linestyle='--')
    ax1.text(best, avg_Kw.max(), 'Best ratio', rotation=20, va='bottom', color = 'red')

    fig.tight_layout()  # otherwise the right y-label is clipped. :/
    plt.show()

def detect_missing_data(data):
    # create a new DataFrame with the expected periods
    expected_periods = pd.date_range(start=data.index.min(), end=data.index.max(), freq='10T')

    # find the missing periods
    missing_periods = expected_periods.difference(data.index)

   # count the number of missing periods for each month
    missing_counts = pd.Series(missing_periods.month).value_counts().sort_index()

    # create a list of missing counts for each month, converted to hours
    # counts = [0] * len(month_names)
    # for i, month in enumerate(month_names):
    #     if month in missing_counts.index:
    #         counts[i] = missing_counts[month] * 0.1667
    counts = [missing_counts[i] * 0.1667 if i in missing_counts.index else 0 for i in range(1, 13)]

    # create a dictionary of month names and their corresponding number
    month_names = [calendar.month_name[i][:3] for i in range(1, 13)]

    # create a bar plot of the missing counts by month
    plt.bar(month_names, counts)
    plt.xlabel('Month')
    plt.ylabel('Missing Data (Hours)')
    plt.title('Missing Data by Month')
    plt.show()

def corr_between_windspeed_activepower_winddirection(data):
    corr_windspeed_power = data["Wind Speed (m/s)"].corr(data["LV ActivePower (kW)"])
    print(corr_windspeed_power)
    corr_winddirection_power = data["Wind Direction (°)"].corr(data["LV ActivePower (kW)"])
    print(corr_winddirection_power)
    corr_windspeed_winddirection = data["Wind Speed (m/s)"].corr(data["Wind Direction (°)"])
    print(corr_windspeed_winddirection)

    # Round wind direction values to nearest 30 degrees
    data["Wind Direction (°) Rounded"] = (data["Wind Direction (°)"] // 30) * 30
    # Group data by wind direction and calculate total active power
    grouped_data = data.groupby("Wind Direction (°) Rounded")["LV ActivePower (kW)"].sum()
    fig, ax = plt.subplots()

    explode = [0.04] * len(grouped_data.index)
    ax.pie(grouped_data.values, labels=grouped_data.index, startangle=90, counterclock=False, explode = explode)
    plt.title("Total Active Power by Wind Direction (°)")
    centre_circle = plt.Circle((0, 0), 0.4, fc='black')
    ax.add_artist(centre_circle)
    plt.savefig("Results/Total_Active_Power_by_Wind_Direction.png")
    plt.show() 
    clearPlts()

    fig, ax = plt.subplots()
    # Group data by wind direction and calculate mean power and wind
    grouped_data = data.groupby("Wind Direction (°) Rounded")["LV ActivePower (kW)", "Wind Speed (m/s)"].mean()
    # Calculate efficiency for each direction

    efficiency = grouped_data["LV ActivePower (kW)"] / grouped_data["Wind Speed (m/s)"]
    
    results = pd.DataFrame({"Efficiency": efficiency})
    print(results.sort_values(by="Efficiency", ascending=False))
    explode = [0.1] * len(grouped_data.index)
    for i in range(len(grouped_data.index)):
        if efficiency.values[i] == efficiency.values.min():
            explode[i] = 0.3

    print(results.sort_values(by="Efficiency", ascending=False))

    ax.pie(results["Efficiency"], labels=results.index, startangle=90, counterclock=False, explode = explode)
    plt.title("Efficiency by Wind Direction (°)")
    centre_circle = plt.Circle((0, 0), 0.4, fc='black')
    ax.add_artist(centre_circle)
    plt.savefig("Results/Efficiency_by_Wind_Direction.png")
    plt.show()

def avg_power_by_windspeed(data):
    # Group data by wind speed and calculate average active power
    #uses the cut function to bin the wind speed values into 3 m/s intervals, and calculates the mean active power for each bin. 
    grouped_data = data.groupby(pd.cut(data["Wind Speed (m/s)"], bins=range(0, 26, 3)))["LV ActivePower (kW)"].mean()

    # Plot bar chart
    plt.bar(grouped_data.index.astype(str), grouped_data.values)
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Average Power Production (kW)")
    plt.title("Average Power Production by Wind Speed")
    plt.grid(color='darkgray')
    # Set x-axis tick labels to display better format
    x_ticks = [f'{bin.left}-{bin.right}' for bin in grouped_data.index]
    plt.xticks(grouped_data.index.astype(str), x_ticks)
    plt.savefig("Results/Average_Power_Production_by_Wind_Speed.png")
    plt.show()

def plot_theoretical_vs_real_power(data):
    # Calculate average power production levels for different wind speeds
    grouped_data = data.groupby(pd.cut(data["Wind Speed (m/s)"], bins=np.arange(0, 25.2, 0.5)))["Wind Speed (m/s)", "Theoretical_Power_Curve (KWh)", "LV ActivePower (kW)"].mean()
    grouped_data.set_index("Wind Speed (m/s)", inplace=True)
   

    fig, ax = plt.subplots()
    ax.plot(grouped_data.index.astype(str), grouped_data["Theoretical_Power_Curve (KWh)"], label="Theoretical Power")
    ax.plot(grouped_data.index.astype(str), grouped_data["LV ActivePower (kW)"], label="LV ActivePower (Avg.)", color='red')
    
    grouped_data['diff'] = grouped_data["LV ActivePower (kW)"] - grouped_data["Theoretical_Power_Curve (KWh)"]


    ax.set_title("Theoretical vs Real Power Production")
    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Power (kW)")
    ax.set_xticks(np.arange(0, len(grouped_data.index), 3))  # set ticks at every 4th index
    ax.set_xticklabels(np.round(grouped_data.index[::3].to_numpy(), 2).astype(str))  # set labels to wind speeds at every 4th index, rounded to 2 decimal places
    ax.legend()
    
    ax2 = ax.twinx()

    # Calculate the number of hours with 0 LV ActivePower (kW) for each wind speed
    data_with_0_power = data[(data["LV ActivePower (kW)"] == 0) & (data["Theoretical_Power_Curve (KWh)"] != 0)]
    hours_with_0_power = data_with_0_power.groupby(pd.cut(data_with_0_power["Wind Speed (m/s)"], bins=np.arange(0, 26, 0.5)))["Wind Speed (m/s)"].count() * 0.1
    grouped_data["Hours with 0 LV ActivePower (kW)"] = hours_with_0_power

    #ax2.plot(grouped_data.index.astype(str), grouped_data["Hours with 0 LV ActivePower (kW)"], label="Hours with 0 LV ActivePower (kW)", color='green')
    plt.savefig("Results/Theoretical_vs_real_power.png")
    plt.show()


def main():
    #data = clean_data(pd.read_csv('wind_turbine_data.csv'))
    data = optimize_dateformat(pd.read_csv('wind_turbine_data.csv'))
    set_dark_bg()
    detect_missing_data(data)
    most_productive_periods(data)
    most_productive_periods(data, 'hour')
    corr_between_windspeed_activepower_winddirection(data)
    avg_power_by_windspeed(data)
    plot_theoretical_vs_real_power(data)

main()