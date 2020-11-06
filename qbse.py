import pandas as pd
import numpy as np
import ast
from tslearn.metrics import dtw

import sys
sys.path.insert(0, './LaurieOnTracking')
import Metrica_IO as mio

# own libraries
import helpers.query as q

match_results = []
matches = ['metrica_1', 'metrica_2']

for cur_match in matches:
    match=cur_match
    # load event data from Metrica always to use reverse function
    DATADIR = './sample-data-master/data/'
    game_id = 1  # let's look at sample match 2

    # read in the event data
    events = mio.read_event_data(DATADIR, game_id)
    if (match == 'metrica_1') or (match == 'metrica_2'):
        if match == 'metrica_1':
            game_id = 1  # let's look at sample match 1
        elif match == 'metrica_2':
            game_id = 2
        DATADIR = './sample-data-master/data/'

        # read in the event data
        events = mio.read_event_data(DATADIR, game_id)

        # Bit of housekeeping: unit conversion from metric data units to meters
        events = mio.to_metric_coordinates(events)
        events.head()
        #### TRACKING DATA ####

        # READING IN TRACKING DATA
        tracking_home = mio.tracking_data(DATADIR, game_id, 'Home')
        tracking_away = mio.tracking_data(DATADIR, game_id, 'Away')

        # Convert positions from metrica units to meters
        tracking_home = mio.to_metric_coordinates(tracking_home)
        tracking_away = mio.to_metric_coordinates(tracking_away)
        # reverse direction of play in the second half so that home team is always attacking from right->left
        tracking_home, tracking_away, events = mio.to_single_playing_direction(tracking_home, tracking_away, events)
        # to fix that home plays to the left direction
        if match == 'metrica_1':
            tracking_home, tracking_away, events = mio.change_playing_direction_both(tracking_home, tracking_away,
                                                                                     events)
        # query features
        query_features = pd.read_csv('./results' + '/metrica_' + str(game_id) +
                                     '_query_features.csv', sep=";")

    # preparation
    nan_bucket = np.array([np.nan] * 8)
    query_features['occupancy map_home'] = [np.array(ast.literal_eval(x)) if pd.notnull(x) else nan_bucket for x in
                                            query_features['occupancy map_home']]
    query_features['occupancy map_away'] = [np.array(ast.literal_eval(x)) if pd.notnull(x) else nan_bucket for x in
                                            query_features['occupancy map_away']]

    query_features['inside_dummy'] = np.where(query_features.inside == True, 1, 0)
    query_features.drop(columns=['inside', 'zone', 'centroid_zone_home',
                                 'centroid_zone_away', 'close to ball'], inplace=True)
    # remove because of too high correlation
    query_features.drop(columns=['rest defence_home', 'centroid x_away', 'centroid y_away'], inplace=True)
    query_features.reset_index()
    query_features.index += 1

    ## normalization
    query_features = q.normalization(query_features)

    # create two types of dataframes, query_diff is only for difference with former frame
    columns = ['ball x', 'ball y', 'centroid x_home', 'centroid y_home', 'width_home', 'length_home',
               'occupancy map_home', 'rest defence_away', 'width_away', 'length_away',
               'occupancy map_away']  # ['centroid y_away', 'centroid x_away', 'rest defence_home']
    query_diff = query_features[columns]
    query_basic = query_features
    # direction important (Absolute differences or not)
    direction = False
    if direction:
        query_diff = query_diff.diff(periods=1)#.abs() # get difference with earlier value
    else:
        query_diff = query_diff.diff(periods=1).abs() # get difference with earlier value
    # merge them together
    #print(query_features.shape[0])
    query_features = pd.merge(query_features, query_diff, left_index=True, right_index=True, suffixes=('', '_diff'))
    query_features = query_features.iloc[1: len(query_features)]  # problems with nan, so remove first obs
    query_features.reset_index()
    #print(query_features.shape[0])

    # aggregate features for episode of size 25*0.4 = 10 seconds
    df_agg = query_features.groupby(query_features.index // 25).agg(
        # df_agg = query_features.groupby('event').agg(
        start_frame=('frame', 'first'),
        end_frame=('frame', 'last'),
        # average positions
        # AVG_ball_x=('ball x', 'mean'),
        # AVG_ball_y=('ball y', 'mean'),
        AVG_ball_vel =('ball velocity', np.nanmean),
        AVG_centroid_x_home=('centroid x_home', 'mean'),
        AVG_centroid_y_home=('centroid y_home', 'mean'),
        # AVG_centroid_x_away=('centroid x_away', 'mean'),
        # AVG_centroid_y_away=('centroid y_away', 'mean'),
        AVG_width_home=('width_home', 'mean'),
        AVG_width_away=('width_away', 'mean'),
        AVG_length_home=('length_home', 'mean'),
        AVG_length_away=('length_away', 'mean'),
        # AVG_rest_defence_home=('rest defence_home', 'mean'),
        AVG_rest_defence_away=('rest defence_away', 'mean'),
        AVG_rom_home=('occupancy map_home', q.rom_avg),
        AVG_rom_away=('occupancy map_away', q.rom_avg),
        AVG_inside=('inside_dummy', 'mean'),
        # trajectories
        SUM_ball_x=('ball x_diff', np.nansum),
        SUM_ball_y=('ball y_diff', np.nansum),
        SUM_centroid_x_home=('centroid x_home_diff', 'sum'),
        SUM_centroid_y_home=('centroid y_home_diff', 'sum'),
        # SUM_centroid_x_away=('centroid x_away_diff', 'sum'),
        # SUM_centroid_y_away=('centroid y_away_diff', 'sum'),
        SUM_width_home=('width_home_diff', 'sum'),
        SUM_width_away=('width_away_diff', 'sum'),
        SUM_length_home=('length_home_diff', 'sum'),
        SUM_length_away=('length_away_diff', 'sum'),
        # SUM_rest_defence_home=('rest defence_home_diff', 'sum'),
        SUM_rest_defence_away=('rest defence_away_diff', 'sum'),
        SUM_rom_home=('occupancy map_home_diff', q.rom_sum),
        SUM_rom_away=('occupancy map_away_diff', q.rom_sum),
        # count
        # count=('rest defence_home', lambda val: (val == 7).sum())
    )
    # metrica 1 is biggest, so take their labels
    if cur_match == 'metrica_1':
        start_frames = df_agg['start_frame']
        end_frames = df_agg['end_frame']

    # Calculate difference euclidean
    # example episode
    if cur_match == 'metrica_1':
        episode_example = df_agg.iloc[97]

    weights = q.set_weights()
    print(weights)
    df_agg['agg_simi'] = [q.episode_difference(episode_example, df_agg.iloc[x], weights) for x in range(len(df_agg))]
    # Calculate difference DTW
    start_frame = df_agg.start_frame.iloc[97]
    end_frame = df_agg.end_frame.iloc[97]
    # same episode in query frame
    if cur_match == 'metrica_1':
        s1 = query_basic[(query_basic.frame >= start_frame) & (query_basic.frame <= end_frame)]
        s1 = s1.drop(columns=['frame', 'occupancy map_home', 'occupancy map_away']).values
    query_basic = query_basic.drop(columns=['frame', 'occupancy map_home', 'occupancy map_away'])
    #calculate dtw
    df_agg['dtw_sim'] = [np.nan] +  [dtw(s1, query_basic.iloc[x:x + 24]) for x in range(25, len(query_basic)-24, 25)] \
                        + [np.nan] # for same length of frames
    match_results.append(df_agg['agg_simi'])
    match_results.append(df_agg['dtw_sim'])
labels = sum([[x + "_eucli", x + "_dtw"] for x in matches], [])

final = pd.DataFrame(match_results).T
final.columns=labels
final['start_frame'] = start_frames
final['end_frame'] = end_frames

#final.to_csv(r'results\similarity_lowblock_away2.csv', index=False, header=True, sep=";")


