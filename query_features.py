import numpy as np
import pandas as pd
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, './LaurieOnTracking')
import Metrica_IO as mio
import Metrica_PitchControl as mpc

# own libraries
import helpers.fundamentals as fund
import helpers.features as feat

# select data
matches = ['metrica_1', 'metrica_2']
match = 'metrica_2'
# pitch & set up initial path to data
if (match == 'metrica_1') or (match == 'metrica_2'):
    if match == 'metrica_1':
        game_id = 1 # let's look at sample match 1
    elif match == 'metrica_2':
        game_id = 2
    DATADIR = './sample-data-master/data/'
    field_dimen = (106.0, 68.0)
    #### TRACKING DATA ####
    # READING IN TRACKING DATA
    tracking_home = mio.tracking_data(DATADIR, game_id, 'Home')
    tracking_away = mio.tracking_data(DATADIR, game_id, 'Away')
    # Convert positions from metrica units to meters
    tracking_home = mio.to_metric_coordinates(tracking_home)
    tracking_away = mio.to_metric_coordinates(tracking_away)
    # reverse direction of play in the second half so that home team is always attacking from right->left

# read in the event data
events = mio.read_event_data(DATADIR,game_id)
# Bit of housekeeping: unit conversion from metric data units to meters
events = mio.to_metric_coordinates(events)
events.head()

tracking_home,tracking_away,events = mio.to_single_playing_direction(tracking_home,tracking_away,events)
# to fix that home plays to the left direction
if match == 'metrica_1':
    tracking_home, tracking_away, events = mio.change_playing_direction_both(tracking_home, tracking_away,
                                                                             events)

# See who's the goalkeepers
GK_numbers = [mio.find_goalkeeper(tracking_home), mio.find_goalkeeper(tracking_away)]
params = mpc.default_model_params()

# field corners
bottomLeft = (-field_dimen[0]/2, -field_dimen[1]/2)
bottomRight = (field_dimen[0]/2, -field_dimen[1]/2)
topLeft = (-field_dimen[0]/2, field_dimen[1]/2)
topRight = (field_dimen[0]/2, field_dimen[1]/2)

# zone of analyst, 18 zones
mapping = list(range(0, 5))[::-1] # mapping to fix weird zone system analysts
cols = np.linspace(bottomLeft[0], bottomRight[0], num=7)
rows = np.linspace(bottomLeft[1], topLeft[1],  num=4)

# Iteration over frames
startframe= 1
endframe =  tracking_home.tail(1).index.values[0] # get last frame
sides = ["Home", "Away"]
storage = {}

# get velocities of the ball
ball_velo_x, ball_velo_y, ball_velo_xy = fund.calc_ball_velocity(tracking_home, "Savitzky-Golay", 7, 1, 200/3.6)

for side in sides:
    if (side == "Home"):
        team = tracking_home
        teamname = side
        GKid = GK_numbers[0]
    elif (side == "Away"):
        team = tracking_away
        teamname = side
        GKid = GK_numbers[1]

    frames = []
    s_coordinates = []
    s_ball_x = []
    s_ball_y = []
    s_ball_velo = []
    s_inside = []
    s_zone = []
    s_centroid_x = []
    s_centroid_y = []
    s_centroid_zone = []
    s_rest_defense = []
    s_width = []
    s_length = []
    s_rom = []
    s_team_distance = []
    s_closest_distance = []

    for frame in range(startframe, endframe, 10):  # 25
        if ((frame-1) % 1500 == 0):
            print("Start calculating for minute " + str(round(frame / 1500)))
        ## Try for the last iteration when not 25 frames
        try:
            coordinates_keep = fund.windowCoordinates(team, teamname, frame, np.nan)
            # remove goalkeeper
            coordinates=coordinates_keep.copy()
            coordinates.pop(GKid)
            coordinates_array = np.array(list(coordinates.values()))
            # coordinates of all frames
            frame_coordinates = np.array([fund.windowCoordinates(team, teamname, frame_x, GKid)
                                          for frame_x in range(frame, frame + 10, 1)])
        except:
            # if change during 25 frame sec or the end
            print(frame)
            s_ball_x.append(np.nan)
            s_ball_y.append(np.nan)
            s_ball_velo.append(np.nan)
            s_inside.append(np.nan)
            s_zone.append(np.nan)
            s_centroid_x.append(np.nan)
            s_centroid_y.append(np.nan)
            s_centroid_zone.append(np.nan)
            s_rest_defense.append(np.nan)
            s_width.append(np.nan)
            s_length.append(np.nan)
            s_rom.append(np.nan)
            s_team_distance.append(np.nan)
            s_closest_distance.append(np.nan)
            frames.append(int(frame))
            continue
        # ball coordinates
        m_ball_x, m_ball_y = fund.ball_coordinates(team, frame)
        m_ball_velo = np.nanmean([np.array(ball_velo_xy.loc[ball_velo_xy.index == frame]) for
                                      x in range(frame, frame + 10)])

        # check if ball mean is inside the pitch
        x_limit = field_dimen[0] / 2
        y_limit = field_dimen[1] / 2

        inside = np.where(np.isnan(m_ball_x), np.nan,
                    np.where(((m_ball_x > -x_limit) & (m_ball_x < x_limit)), \
                                    np.where(((m_ball_y > -y_limit) & (m_ball_y < y_limit)),
                                                    1, 0), 0))
        inside=np.where(inside==1, True, False)
        # find region
        if inside:
            ball_zone_x = np.searchsorted(cols, m_ball_x)
            ball_zone_y = mapping.index(np.searchsorted(rows, m_ball_y))
            ball_zone = ball_zone_y + (ball_zone_x - 1) * 3
            closest_distance = next(iter(feat.distance_to_ball(m_ball_x, m_ball_y, coordinates_keep).values()))
        else:
            ball_zone = np.nan
            closest_distance = np.nan

        # centroid
        m_centroid_x, m_centroid_y = fund.team_centroid(coordinates_array)
        centroid_zone_x = np.searchsorted(cols, m_centroid_x)
        centroid_zone_y = mapping.index(np.searchsorted(rows, m_centroid_y))
        centroid_zone = centroid_zone_y + (centroid_zone_x - 1) * 3

        # Rest defense home team
        m_rest_defense = feat.rest_defense(coordinates_array[:, 0], m_ball_x, teamname)

        # width
        width_max_player_x, width_max_player_y = fund.extrema_axis_player(coordinates_array, extrema='max', axis='y-axis')
        width_min_player_x, width_min_player_y = fund.extrema_axis_player(coordinates_array, extrema='min', axis='y-axis')
        width = width_max_player_y - width_min_player_y

        # length change
        length_max_player_x, length_max_player_y = fund.extrema_axis_player(coordinates_array, extrema='max', axis='x-axis')
        length_min_player_x, length_min_player_y = fund.extrema_axis_player(coordinates_array, extrema='min', axis='x-axis')
        length = length_max_player_x - length_min_player_x

        # relative occupancy map
        if np.isnan(m_ball_x):
            rom = np.nan
        else:
            rom = [0] * 8
            [feat.rom_zone(rom, m_ball_x, m_ball_y, coordinate[0], coordinate[1]) for coordinate in coordinates.values()]

        # player distances trajectory
        team_distance = feat.sum_team_distance(frame_coordinates)

        # push in buckets
        #s_coordinates.append(coordinates)
        s_ball_x.append(m_ball_x)
        s_ball_y.append(m_ball_y)
        s_ball_velo.append(m_ball_velo)
        s_inside.append(inside)
        s_zone.append(ball_zone)
        s_centroid_x.append(m_centroid_x)
        s_centroid_y.append(m_centroid_y)
        s_centroid_zone.append(centroid_zone)
        s_rest_defense.append(m_rest_defense)
        s_width.append(width)
        s_length.append(length)
        s_rom.append(rom)
        s_team_distance.append(team_distance)
        s_closest_distance.append(closest_distance)
        frames.append(int(frame))

        # Store in DataFrame
    labels = ["frame", "ball x", "ball y", "ball velocity", "inside", "zone", "centroid x", "centroid y",
              "centroid_zone", "rest defence", "width", "length", "occupancy map", "team distance", "closest to ball"]
    storage[side] = pd.DataFrame(
            list(zip(frames, s_ball_x, s_ball_y, s_ball_velo, s_inside, s_zone,  s_centroid_x,
                     s_centroid_y, s_centroid_zone, s_rest_defense, s_width, s_length, s_rom, s_team_distance, s_closest_distance)),  columns=labels)

home = storage["Home"]
away = storage["Away"]

# drop ball columns to prevent duplicate columsn
away.drop(columns=["frame", "ball x", "ball y", 'ball velocity', "inside", "zone"], inplace=True)
#
print("before home " + str(home.shape))
print("before away " + str(away.shape))

final = home.merge(away, left_index=True, right_index=True,
          suffixes=('_home', '_away'))
# ball out or distance to ball greater than 3 > nan
final['close to ball'] = np.where(pd.isnull(final['closest to ball_home']) | ((final['closest to ball_away'] > 3) &
                                  (final['closest to ball_home'] > 3)), np.nan,
                                np.where(final['closest to ball_home'] < final['closest to ball_away'], "Home", "Away"))
final.drop(columns=['closest to ball_home', 'closest to ball_away'], inplace=True)
print("after final " + str(final.shape))

# Save to csv
final.to_csv(r'results\metrica_2_query_features.csv', index=False, header=True, sep=";")