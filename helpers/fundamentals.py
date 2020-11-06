import numpy as np
from scipy.spatial import ConvexHull
import scipy.signal as signal

def get_coordinates(team, teamname, frame=1, GKid=np.nan):
    """ Retrieve the coordinates for the provided team and frame
    Args:
        team: row (i.e. instant) of the team tracking data frame
        teamname: team name "Home" or "Away"
        frame: Integer representing the frame number for the calculation
        GKid: When to exclude goalkeeper, provide id of the goalkeepr

    Returns:
        coordinates: List with the coordinates of the players
    """

    # Player id's in the frame
    player_ids = np.unique([c.split('_')[1] for c in team.keys() if c[:4] == teamname])

    # Goalkeeper is removed when GKid is provided
    if GKid != np.nan:
        player_ids = np.delete(player_ids, np.where(player_ids == GKid))

    # Get frame
    team_frame = team[(team.index == frame)]

    # Get the coordinates of the players
    coordinates = {player_id: [team_frame[teamname + "_" + player_id + "_x"].values[0],
                               team_frame[teamname + "_" + player_id + "_y"].values[0]] for player_id in player_ids}

    # remove players with nan a x-coordinate
    coordinates = {k: v for k, v in coordinates.items() if not np.isnan(v[0])}
    return coordinates


def windowCoordinates(team, teammname, frame, GKid):
    """ Retrieve the mean values for a 0.4s (10 frames)
    Args:
        team: row (i.e. instant) of the team tracking data frame
        teamname: team name "Home" or "Away"
        frame: Integer representing the frame number for the calculation

    Returns:
        coordinates: dictionary with key player id, and value mean position
    """

    frame_coordinates = [get_coordinates(team, teammname, x, GKid) for x in range(frame, frame + 10)]
    num_players = frame_coordinates[0]  # first item
    coordinates = {k: np.mean([frame[k] for frame in frame_coordinates], axis=0) for k in num_players.keys()}
    return coordinates


def ball_coordinates(team, frame):
    """ Retrieve the mean values for a 1s (25 frames)
    Args:
        frame: Integer representing the frame number for start of the calculation
    Returns:
        ball_x : Mean x coordinate of the ball
        ball_y : mean y coordinate of the ball

    """
    ball_x = np.nanmean([np.array(team.loc[team.index == x, ['ball_x']].values[0][0]) for
                         x in range(frame, frame + 10)])
    ball_y = np.nanmean([np.array(team.loc[team.index == x, ['ball_y']].values[0][0]) for
                         x in range(frame, frame + 10)])
    return ball_x, ball_y

def gen_convex_hull(coordinates):
    """ Generate convex hull
    Function that generates the convex hull
    Args:
        coordinates: List with the coordinates of the players that are used to calculate the convex hull
    Returns
        hull: scipy.spatial.qhull.ConvexHull that represents the convex hull
    """

    # create convex hull
    hull = ConvexHull(coordinates)
    return hull

def team_centroid(coordinates):
    """ Calculates the team centroid of a list of coordinates

    Args:
        coordinates: List with the coordinates of the players that are used for calculation
    Returns:
        x_centroid: A float that represents the centroid with respect to x-axis
        y_centroid: A float that represents the centroid with respect to y-axis
    """
    x_centroid = np.mean([coordinate[0] for coordinate in coordinates])
    y_centroid = np.mean([coordinate[1] for coordinate in coordinates])

    return x_centroid, y_centroid

def extrema_axis_player(coordinates, extrema='max', axis='y-axis'):
    """ Finds the coordinates of a player for the selected extrema and axis
    Args:
        coordinates: List with the coordinates of the players that are used for calculation
        extrema: either 'max' or 'min' indicating the extrema to take
        axis: either 'y-axis' or 'x-axis' indicating the axis to perform the action
    """
    # Correct for selected axis
    if axis == 'y-axis':
        axis_coordinates = [coordinate[1] for coordinate in coordinates]
    elif axis == 'x-axis':
        axis_coordinates = [coordinate[0] for coordinate in coordinates]

    # Correct for selected extrema
    if extrema == 'max':
        extrema_player = np.where(axis_coordinates == np.amax(axis_coordinates))[0][0]
    elif extrema == 'min':
        extrema_player = np.where(axis_coordinates == np.amin(axis_coordinates))[0][0]
    x_extrema = coordinates[extrema_player][0]
    y_extrema = coordinates[extrema_player][1]
    # distance = abs(maximum-minimum)
    return x_extrema, y_extrema


def calc_ball_velocity(team, smoothing=True, filter_='Savitzky-Golay', window=7, polyorder=1, maxspeed=200 / 3.6):
    """  credits to Lauri Shaw (https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking)
    Calculate ball velocity in x & y direction, and total ball speed at each timestamp of the tracking data
    Parameters
    -----------
        team: the tracking DataFrame for home or away team
        smoothing: boolean variable that determines whether velocity measures are smoothed. Default is True.
        filter: type of filter to use when smoothing the velocities. Default is Savitzky-Golay, which fits a polynomial of order 'polyorder' to the data within each window
        window: smoothing window size in # of frames
        polyorder: order of the polynomial for the Savitzky-Golay filter. Default is 1 - a linear fit to the velcoity, so gradient is the acceleration
        maxspeed: the maximum speed that a player can realisitically achieve (in meters/second). Speed measures that exceed maxspeed are tagged as outliers and set to NaN.

    Returrns
    -----------
       ball_velo_x : Speed in x direction
       ball_velo_y : Speed in y direction
       ball_velo   : Total speed of the ball

    """
    # index of first frame in second half
    second_half_idx = team.Period.idxmax(2)

    # Calculate the timestep from one frame to the next. Should always be 0.04 within the same half
    dt = team['Time [s]'].diff()

    # difference player positions in timestep dt to get unsmoothed estimate of velicity
    vx = team["ball_x"].diff() / dt
    vy = team["ball_y"].diff() / dt

    if maxspeed > 0:
        # remove unsmoothed data points that exceed the maximum speed (these are most likely position errors)
        raw_speed = np.sqrt(vx ** 2 + vy ** 2)
        vx[raw_speed > maxspeed] = np.nan
        vy[raw_speed > maxspeed] = np.nan

        if smoothing:
            if filter_ == 'Savitzky-Golay':
                # calculate first half velocity
                vx.loc[:second_half_idx] = signal.savgol_filter(vx.loc[:second_half_idx], window_length=window,
                                                                polyorder=polyorder)
                vy.loc[:second_half_idx] = signal.savgol_filter(vy.loc[:second_half_idx], window_length=window,
                                                                polyorder=polyorder)
                # calculate second half velocity
                vx.loc[second_half_idx:] = signal.savgol_filter(vx.loc[second_half_idx:], window_length=window,
                                                                polyorder=polyorder)
                vy.loc[second_half_idx:] = signal.savgol_filter(vy.loc[second_half_idx:], window_length=window,
                                                                polyorder=polyorder)
            elif filter_ == 'moving average':
                ma_window = np.ones(window) / window
                # calculate first half velocity
                vx.loc[:second_half_idx] = np.convolve(vx.loc[:second_half_idx], ma_window, mode='same')
                vy.loc[:second_half_idx] = np.convolve(vy.loc[:second_half_idx], ma_window, mode='same')
                # calculate second half velocity
                vx.loc[second_half_idx:] = np.convolve(vx.loc[second_half_idx:], ma_window, mode='same')
                vy.loc[second_half_idx:] = np.convolve(vy.loc[second_half_idx:], ma_window, mode='same')
    ball_velo_x = vx
    ball_velo_y = vy
    ball_velo = np.sqrt(vx ** 2 + vy ** 2)
    return ball_velo_x, ball_velo_y, ball_velo


