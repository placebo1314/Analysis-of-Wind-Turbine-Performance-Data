import calendar
from datetime import timedelta
from colorama import Fore
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

    # Distribution of the data
    sns.displot(data['LV ActivePower (kW)'])

    # Correlation matrix
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True)

def optimize_dateformat(data):
    data["Date"] = pd.to_datetime(data['Date/Time'], format='%d %m %Y %H:%M')
    data.drop('Date/Time', axis = 1, inplace = True)
    data.set_index("Date", inplace = True)
    return data

def anomaly_detector(data):
    from sklearn.ensemble import IsolationForest

    # Train the isolation forest algorithm
    isolation_forest = IsolationForest(n_estimators=100, contamination=0.002)
    isolation_forest.fit(data)

    # Predict anomalies in the dataset
    is_anomaly = isolation_forest.predict(data)
    data['anomaly'] = is_anomaly
    data['scores'] = isolation_forest.decision_function(data.drop('anomaly', axis=1))  # exclude the 'anomaly' column

    # Print anomalies in the data
    anomaly_data = data[data['anomaly'] == -1]
    #pd.set_option("display.max_rows", None, "display.max_columns", None)

    # Plot anomalies in the data
    plt.plot(data[data.index.month == 1].index, data[data.index.month == 1]['LV ActivePower (kW)'], color='blue')
    plt.plot(data[data.index.month == 1].index, data[data.index.month == 1]['Wind Speed (m/s)']*100, color='red')
    plt.scatter(anomaly_data[anomaly_data.index.month == 1].index, anomaly_data[anomaly_data.index.month == 1]['LV ActivePower (kW)'], color='purple')
    plt.scatter(anomaly_data[anomaly_data.index.month == 1].index, anomaly_data[anomaly_data.index.month == 1]['Wind Speed (m/s)'], color='green')
    plt.show()

def check_wind_trends(data):
    data = data[data.index.month == 1]
    plt.plot(data.index, data['Wind Speed (m/s)'], label = data.index, linewidth = 1, color = 'red')
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    plt.show()

def anomalies_Kw(data):
    # calculate absolute difference between columns
    data['diff_Kw'] = (data['LV ActivePower (kW)'] - data['Theoretical_Power_Curve (KWh)'])
    # Calculate the mean and standard deviation of power_diff
    mean_diff = data['diff_Kw'].mean()
    std_diff = data['diff_Kw'].std()

    # Calculate the threshold for the anomaly detection
    pos_threshold = 200
    neg_threshold = mean_diff - 3 * std_diff

    # Create a new DataFrame with only the anomalies greater than the threshold
    pos_anomalies_df = data[(data['diff_Kw'] > pos_threshold) & (data['LV ActivePower (kW)'] != 0)]
    neg_anomalies_df = data[(data['diff_Kw'] < neg_threshold) & (data['LV ActivePower (kW)'] != 0)]
    zero_anomalies_df = data[data['LV ActivePower (kW)'] == 0]

    data["breakdown"] = 0
    # for i in range(len(data)):
    #     if (data['LV ActivePower (kW)'][i] == 0) & (data['Theoretical_Power_Curve (KWh)'][i] != 0):
    #             print(f"\n{Fore.RED}anomaly:" + "\033[0m")
    #             print(data.iloc[i])
            #print(data.iloc[i])
    minwind = data[(data['LV ActivePower (kW)'] == 0) & (data['Theoretical_Power_Curve (KWh)'] != 0) & (data['Wind Speed (m/s)'] > 4)]
    print(minwind.describe())
 
    # breakdowns = zero_anomalies_df[zero_anomalies_df['Theoretical_Power_Curve (KWh)'] == 0]
    # #print(breakdowns)
    # breakdown_intervals = []
    # for i in range(len(breakdowns)):
    #     if i == 0:  # if first row, set the start time
    #         start_time = breakdowns.index[i]
    #     else:
    #         time_diff = breakdowns.index[i] - breakdowns.index[i-1]
    #         if time_diff == timedelta(minutes=10):  # if the time difference is 10 minutes, continue the interval
    #             continue
    #         else:  # if the time difference is not 10 minutes, end the current interval and start a new one
    #             end_time = breakdowns.index[i-1]
    #             breakdown_intervals.append((start_time, end_time))
    #             start_time = breakdowns.index[i]
    # # add the last interval to the list
    # end_time = breakdowns.index[-1]
    # breakdown_intervals.append((start_time, end_time))

    # # create a DataFrame from the breakdown intervals list
    # breakdowns_df = pd.DataFrame(breakdown_intervals, columns=['start_time', 'end_time'])
    # # calculate the duration of each breakdown in minutes
    # breakdowns_df['duration'] = (breakdowns_df['end_time'] - breakdowns_df['start_time']).dt.total_seconds() / 60
    # # group the breakdowns by hour and calculate the average number of 10-minute breakdowns per hour
    # breakdowns_df['hour'] = breakdowns_df['start_time'].dt.hour
    # print(breakdowns_df)
    # breakdowns_by_hour = breakdowns_df.groupby('hour').agg(avg_breakdowns=('duration', lambda x: len(x[x >= 10]) / (60 / 10)))

    # # create a bar plot of the average number of breakdowns per hour
    # breakdowns_by_hour.plot(kind='bar', y='avg_breakdowns', rot=0)
    # plt.xlabel('Hour')
    # plt.ylabel('Average number of 10-minute breakdowns')
    # plt.show()





    # for i in breakdown_intervals:
    #     print(i)

    # plt.plot(data['LV ActivePower (kW)'])
    # for interval in breakdown_intervals:
    #     plt.axvspan(interval[0], interval[1], color='red', alpha=0.3)
    # plt.show()

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.scatter(pos_anomalies_df.index, pos_anomalies_df['diff_Kw'], c='g', s=10)
    plt.scatter(neg_anomalies_df.index, neg_anomalies_df['diff_Kw'], c='r', s=10)
    plt.scatter(zero_anomalies_df.index, zero_anomalies_df['diff_Kw'], c='black', s=10)
    plt.axhline(y=mean_diff, color='k', linestyle='--', linewidth=1)
    plt.axhline(y=pos_threshold, color='g', linestyle='--', linewidth=1)
    plt.axhline(y=neg_threshold, color='r', linestyle='--', linewidth=1)
    plt.title('Difference between LV ActivePower (kW) and Theoretical_Power_Curve (KWh)')
    plt.xlabel('Time')
    plt.ylabel('Difference')
    #plt.show()

def lost_wind(data):
    minwind = data[(data['LV ActivePower (kW)'] == 0) & (data['Theoretical_Power_Curve (KWh)'] != 0) & (data['Wind Speed (m/s)'] > 4)]
    #plt.plot(data.index, data['Wind Speed (m/s)'], label = data.index, linewidth = 1, color = 'green')
    fig, ax = plt.subplots(figsize=(10, 6))
    mask = data.index.isin(minwind.index)
    # Plot LV ActivePower (kW) with red line for periods where value is 0
    plt.plot(data['Wind Speed (m/s)'].where(data['LV ActivePower (kW)']!=0), color='blue')
    plt.plot(data['Wind Speed (m/s)'].where(data['LV ActivePower (kW)']==0), color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Wind Speed (m/s)')
    ax.set_title('Wind Speed vs. Date')

    #make scroll:
    axcolor = 'lightgoldenrodyellow'
    axpos = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)
    slider = Slider(axpos, 'Date', data.index[0].timestamp(), data.index[-1].timestamp(), valinit=data.index[0].timestamp())

    def update(val):
        pos = pd.Timestamp.fromtimestamp(slider.val, tz=data.index.tz)
        ax.set_xlim(pos, pos + pd.Timedelta(hours=1))
        # Format x-axis tick labels
        date_format = dates.DateFormatter('%Y-%m-%d %H:%M')
        ax.xaxis.set_major_formatter(date_format)
        ax.xaxis.set_tick_params(rotation=45)
        ax.xaxis_date()
        plt.gcf().autofmt_xdate()
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()





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
     # Add a vertical line at the highest product value
    ax1.axvline(x=best, color='red', linestyle='--')

    fig.tight_layout()  # otherwise the right y-label is clipped. :/
    plt.show()

def detect_missing_data(data):
    # create a new DataFrame with the expected periods
    expected_periods = pd.date_range(start=data.index.min(), end=data.index.max(), freq='10T')

    # find the missing periods
    missing_periods = expected_periods.difference(data.index)
    # print the number of missing periods and the missing periods themselves
    #print(f"Number of missing periods: {len(missing_periods)}")
    # #check Feb.:
    # pd.set_option("display.max_rows", None, "display.max_columns", None)
    # febr = data[data.index.month == 2]
    # print(febr.describe())
    # #check january:
    # missing_records = 4464 - len(data[data.index.month == 1])
    # missing_hours = missing_records / 6
    # print(missing_hours)

   # count the number of missing periods for each month
    missing_counts = pd.Series(missing_periods.month).value_counts().sort_index()

    # create a list of missing counts for each month, converted to hours
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
    corr_windspeed_power = data["Wind Direction (°)"].corr(data["LV ActivePower (kW)"])
    print(corr_windspeed_power)

    # Round wind direction values to nearest 10 degrees
    data["Wind Direction (°) Rounded"] = (data["Wind Direction (°)"] // 30) * 30
    
    # Group data by wind direction and calculate total active power
    grouped_data = data.groupby("Wind Direction (°) Rounded")["LV ActivePower (kW)"].sum()
    
    # Create pie chart
    plt.pie(grouped_data.values, labels=grouped_data.index, startangle=90, counterclock=False)
    plt.title("Total Active Power by Wind Direction (°)")
    clearPlts()
    #plt.show()

    # Group data by wind direction and calculate mean power and wind
    grouped_data = data.groupby("Wind Direction (°) Rounded")["LV ActivePower (kW)", "Wind Speed (m/s)"].mean()
    # Calculate efficiency for each direction
    efficiency = grouped_data["LV ActivePower (kW)"] / grouped_data["Wind Speed (m/s)"]

    results = pd.DataFrame({"Efficiency": efficiency})
    print(results.sort_values(by="Efficiency", ascending=False))

    plt.pie(results["Efficiency"], labels=results.index, startangle=90, counterclock=False)
    plt.title("Efficiency by Wind Direction (°)")
    plt.show()

def avg_power_by_windspeed(data):
    # Group data by wind speed and calculate average active power
    #uses the cut function to bin the wind speed values into 5 m/s intervals, and calculates the mean active power for each bin. 
    grouped_data = data.groupby(pd.cut(data["Wind Speed (m/s)"], bins=range(0, 26, 5)))["LV ActivePower (kW)"].mean()

    # Plot bar chart
    plt.bar(grouped_data.index.astype(str), grouped_data.values)
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Average Power Production (kW)")
    plt.title("Average Power Production by Wind Speed")
    plt.show()

def plot_theoretical_vs_real_power(data):
    # Calculate average power production levels for different wind speeds
    gruped_data = data.groupby(data["Wind Speed (m/s)"])["Theoretical_Power_Curve (KWh)", "LV ActivePower (kW)"].mean()
    fig, ax = plt.subplots()
    #print(gruped_data)
    ax.plot(gruped_data.index, gruped_data["Theoretical_Power_Curve (KWh)"], label="Theoretical Power")

    gruped_data['diff'] = gruped_data["LV ActivePower (kW)"] - gruped_data["Theoretical_Power_Curve (KWh)"]

    # Add scatter points for LV ActivePower (kW) values that differ from the theoretical power production curve
    real_power = gruped_data["LV ActivePower (kW)"]
    theoretical_power = gruped_data["Theoretical_Power_Curve (KWh)"]
    diff = real_power - theoretical_power
    print(diff)
    ax.scatter(gruped_data.index[diff < 0], real_power[diff < 0], color='orange', label="Actual power < Theoretical power")
    ax.scatter(gruped_data.index[diff > 0], real_power[diff > 0], color='green', label="Actual power > Theoretical power")
    # Set plot title and axis labels
    ax.set_title("Theoretical vs Real Power Production")
    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Power (kW)")

    # Add legend and display plot
    ax.legend()
    plt.show()

def main():
    #data = clean_data(pd.read_csv('wind_turbine_data.csv'))
    data = optimize_dateformat(pd.read_csv('wind_turbine_data.csv'))
    set_dark_bg()
    #detect_missing_data(data)
    # most_productive_periods(data)
    # most_productive_periods(data, 'hour')
    #corr_between_windspeed_activepower_winddirection(data)
    #avg_power_by_windspeed(data)
    plot_theoretical_vs_real_power(data)

main()