import numpy as np
import pandas as pd

def find_intervals(query_interval, condition):
    """ Method to find intervals that meets the condition
    params:
        query_interval: dataframe with feature values
        condition: list with true/false values for on query_interval for a certain condition

    returns:
        intervals: list with intervals that meet the condition
    """
    query_interval['cond'] = condition
    query_interval['phase'] = (query_interval['cond'] != query_interval['cond'].shift(1)).fillna(0).cumsum(skipna=False)
    phases = query_interval.groupby('phase').agg(
        Condition=('cond', 'first'),
        Start_frame=('frame', 'first'),
        End_frame=('frame', 'last')
    )
    phases = phases[phases.Condition]
    intervals = [range(phases.Start_frame.iloc[x], phases.End_frame.iloc[x] + 40) for x in range(len(phases))]
    return intervals


def find_intersections(intervalsA, intervalsB):
    """Function to find which intervals that intersect with each other
    params:
        intervalsA: list with ranges
        intervalsB: list with ranges
    returns:
        intersections: a list of ranges from intervalsB that intersect with intervalS
    """
    intersections = []
    i = j = 0
    value = np.nan
    while i < len(intervalsA) and j < len(intervalsB):

        # a before b, otherwise increase j
        #if min(intervalsA[i]) > min(intervalsB[j]):
        #    j +=1
        #    continue

        # Left bound for intersecting segment / biggest start
        left = max(min(intervalsA[i]), min(intervalsB[j]))
        # Right bound / smallest end point
        right = min(max(intervalsA[i]), max(intervalsB[j]))

        if left <= right:
            if intervalsB[j] != value:
                intersections.append(intervalsB[j])
                value = intervalsB[j]

        # If i-th interval's right bound is
        # smaller increment i else increment j
        if max(intervalsA[i]) < max(intervalsB[j]):
            i += 1
        else:
            j += 1

    return intersections


def check_results(test, conditions):
    """ Functions that checks which episodes meet the conditions
    Params:
        test: DF with all features
        conditions: listed with conditions in formation [["A", query.inside==False], ...]
    returns:
        results: list with all episodes that meet the conditions
    """
    # Check conditions and put letter in query DF
    interval_combi = find_intervals(test, conditions[0])

    for num_cond in range(1, len(conditions)):
        interval = find_intervals(test, conditions[num_cond])
        interval_combi = find_intersections(interval_combi, interval)
    return interval_combi

def normalization(query_features):
    """ Normalize columns with min max normalization using artificial values for min and max
    Params:
        query_features: df with columns as specified here
    Returns:
        normalized query_features
    """
    # ball x
    ball_x_min = -55
    ball_x_max = 55
    query_features['ball x'] = (query_features['ball x'] - ball_x_min) / (ball_x_max - ball_x_min)
    # ball y
    ball_y_min = -37
    ball_y_max = 37
    query_features['ball y'] = (query_features['ball y'] - ball_y_min) / (ball_y_max - ball_y_min)
    query_features['ball y'].min()
    # centroid x
    query_features['centroid x_home'] = (query_features['centroid x_home'] - ball_x_min) / (ball_x_max - ball_x_min)
    # query_features['centroid x_away'] = (query_features['centroid x_away'] - ball_x_min)/(ball_x_max-ball_x_min)
    # centroid y
    query_features['centroid y_home'] = (query_features['centroid y_home'] - ball_y_min) / (ball_y_max - ball_y_min)
    # query_features['centroid y_away'] = (query_features['centroid y_away'] - ball_y_min)/(ball_y_max-ball_y_min)
    # length
    length_max = 110
    length_min = 0
    query_features['length_home'] = (query_features['length_home'] - length_min) / (length_max - length_min)
    query_features['length_away'] = (query_features['length_away'] - length_min) / (length_max - length_min)
    # width
    width_max = 74
    width_min = 0
    query_features['width_home'] = (query_features['width_home'] - width_min) / (width_max - width_min)
    query_features['width_away'] = (query_features['width_away'] - width_min) / (width_max - width_min)
    # rom
    query_features['occupancy map_home'] = query_features['occupancy map_home'] / 8
    query_features['occupancy map_away'] = query_features['occupancy map_away'] / 8
    # team distance
    distance_max = 8.6 * 0.4 * 10  # average top speed * time frame * num_players
    distance_min = 0
    query_features['team distance_home'] = (query_features['team distance_home'] - distance_min) / (
            distance_max - distance_min)
    query_features['team distance_away'] = (query_features['team distance_away'] - distance_min) / (
            distance_max - distance_min)
    # ball velocity
    velo_max = 200 / 3.6
    velo_min = 0
    query_features['ball velocity'] = query_features['ball velocity'] / velo_max
    # rest defence_away
    rest_defence_max = 10
    query_features['rest defence_away'] = query_features['rest defence_away'] / rest_defence_max
    return query_features


def rom_sum(series):
    """ Calculate sum for a series of values for ROM
    Params:
        series
    return:
    """
    nan_bucket = np.array([np.nan] * 8)
    try:
        return [np.array(np.nansum([series], axis=1))]
    except:
        return nan_bucket

def rom_avg(series):
    """ Calculate mean for a aseries of values for ROM
    Params:
        series
    Returns

    """
    nan_bucket = np.array([np.nan] * 8)
    #return str(reduce(lambda x, y: x + y, series)/len(series))
    try:
        return [np.array(np.nanmean([series], axis=1)[0])]
    except:
        return nan_bucket


def episode_difference(episodeA, episodeB, weights):
    """ Function to calculate difference between episodes
    params:
        df_agg: dataframe with aggregate values for every episode
        episodeA: integer indicating the first episode
        episodeB: integer indicating the first episode
        weight: numpy array with weights summing up to 1
    """
    rom_weight = weights[-1]
    weights_nonrom = weights[:-1]
    simple_columns = ['start_frame', 'end_frame', 'AVG_rom_home', 'AVG_rom_away', 'SUM_rom_home', 'SUM_rom_away']
    rommies = ['AVG_rom_home', 'AVG_rom_away', 'SUM_rom_home', 'SUM_rom_away']
    df_agg_simple_A = episodeA.drop(index = simple_columns)
    df_agg_simple_B = episodeB.drop(index = simple_columns)
    simple_difference = np.nansum(weights_nonrom*(np.absolute(df_agg_simple_A - df_agg_simple_B)))
    rom_difference = rom_weight * np.nansum(
        [np.linalg.norm(episodeA[rom][0] - episodeB[rom][0]) for rom in rommies])
    return simple_difference + rom_difference

def set_weights():
    """
    Weights must be 0 and 1
    """
    return np.array([
        0.052,   #AVG_ball_vel
        0.052,    #'AVG_centroid_x_home'
        0.052,    #'AVG_centroid_y_home'
        0.052,    #'AVG_width_home'
        0.052,    #'AVG_width_away',
        0.052,     # AVG_length_home
        0.052,     # AVG_length_away
        0.052,    #'AVG_rest_defence_away'
        0.052,    #'AVG_inside',
        0.052,    #'SUM_ball_x',
        0.052,     #'SUM_ball_y
        0.052,    #'SUM_centroid_x_home',
        0.052,    #'Sum centroid y_home
        0.052,    #'SUM_width_home',
        0.052,    # SUM_width_away',
        0.052,    # SUM_length_home
        0.052,    # SUM_length_away
        0.052,    # 'SUM_rest_defence_away'
        0.052,   # roms
    ])

