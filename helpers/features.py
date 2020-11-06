
from scipy.spatial import distance
import numpy as np
from numpy import linalg as LA
import math
import helpers.fundamentals as fund


def area_convex_hull(hull):
    """ Calculates area of convex hull on the pitch

    Args:
        hull: scipy.spatial.qhull.ConvexHull that represents the convex hull

    Returns:
        hull.area: a float with area of the convex hull
    """
    return hull.area

# Todo check
def team_spread(coordinates):
    """ Calculate the team spread using the method of Moura et al. (2012)

    Args:
        coordinates: List with the coordinates of the players that are used for calculation

    returns:
        frob_norm: A float representing the team spread

    """
    # Calculates the distances between all players
    distances_matrix = distance.cdist(coordinates, coordinates, 'euclidean')
    # get lower triangular matrix
    distances_matrix = np.tril(distances_matrix, k=0)
    # Frobenius norm for matrix
    frob_norm = LA.norm(distances_matrix, 'fro')
    return frob_norm

def stretch_index(coordinates):
    """ Calculate the stretch index using the method of Duarte et al. (2013)

    Args:
        coordinates: List with the coordinates of the players that are used for calculation

    returns:
        stretch_index: A float representing the team spread

    """
    # Calculate the coordinates of the centroid
    x_centroid, y_centroid = fund.team_centroid(coordinates)
    # Calculate the stretch index
    stretch_index = np.mean([distance.euclidean([x_centroid, y_centroid], [coordinate[0], coordinate[1]]) for coordinate
                             in coordinates])
    return stretch_index

def rest_defense(x_coordinates, x_ball, teamname='Home'):
    """ Counts the numbers of players behind the ball in a selected frame

    Args:
        x_coordinates: A list of x-coordinates of the players in a selected frame
        x_ball: the x-coordinate of the ball
        side: boolean indicting if the team coordinates are from the home of away team

    returns:
        num_players: The number of players behind the ball

    """

    # Return nan if ball coordinates are unavailable
    if np.isnan(x_ball):
        return np.nan

    if (teamname == 'Home'):
        num_players = sum(1 if x > x_ball else 0 for x in x_coordinates)
    else:
        num_players = sum(1 if x < x_ball else 0 for x in x_coordinates)
    return num_players


def sum_team_distance(frame_coordinates):
    """ Retrieve sum of all the euclidean distances by a player for a selected window
    Args:
        frame_coordinates: the coordinates of the player for every frame in the window for a selected team

    Returns:
       team_distance: Float represing the walked distance of the team
    """
    # Get the sum of the euclidean distances  for all players
    team_distance = np.sum([  # Sum all individual player distances
        np.sum([  # Sum all distances between frames
            np.linalg.norm(  # calculate distance between two frames
                frame_coordinates[frame][player_id] - frame_coordinates[frame+1][player_id])
            for frame in range(len(frame_coordinates) - 1)])  # number of frames - 1
        for player_id, coordinate in frame_coordinates[0].items()]) # number of players
    return team_distance


def distance_to_ball(ball_x, ball_y, coordinates):
    """ Retrieve the euclidean distances to the ball for all players, nan if ball is outside the pitch
    Args:
        :param x_ball: x coordinate of the ball
         :param y_ball: y coordinate of the ball
        coordinates: sorted dictionary with player id and distance to the ball

    Returns:
        distances: sorted dictionary of all players with distance
    """
    # calculate distances
    distances = {player_id: np.linalg.norm(np.array(coordinate) - np.array([ball_x, ball_y])) for player_id, coordinate
                 in coordinates.items()}
    # sort
    distances = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
    return distances

def rom_zone(buckets, x_ball, y_ball, x_player, y_player):
    """ Find the zone in the relative occupancy map with a radius of 9 meter and increments the bucket belonging to the
    zone with one
    :param buckets: An array wist 8 values for every zone
    :param x_ball: x coordinate of the ball
    :param y_ball: y coordinate of the ball
    :param x_player: x coordinate of the player
    :param y_player: y coordinate of the player
    :return: buckets: An array wist 8 values for every zone, the zone of the player is increment with one
    """
    distance = np.linalg.norm(np.array([x_player, y_player]) - np.array([x_ball, y_ball]))
    angle = math.degrees(math.atan2((y_player-y_ball), (x_player-x_ball)))
    zone = np.where(distance > 9,  4, 0)
    zone = np.where((angle > -45) & (angle < 135),
                   np.where(angle < 45, zone, zone+1),
                   np.where((angle < -135) | (angle > 135), zone+2, zone+3)
                   )
    buckets[zone] +=1
    return buckets


