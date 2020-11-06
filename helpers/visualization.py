import sys

# credits to Lauri Shaw (https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking)
sys.path.insert(0, '../.././data/code/LaurieOnTracking')
import Metrica_Viz as mviz
import helpers.fundamentals as fund


def vis_convex_hull(hometeam, awayteam, coordinates, hull, frame=1):
    """ Visualizes convex hull on the pitch

    Args:
        hometeam: row (i.e. instant) of the home team tracking data frame
        awayteam: row of the away team tracking data frame
        coordinates: List with the coordinates of the players that are used to calculate the convex hull
        hull: scipy.spatial.qhull.ConvexHull that represents the convex hull
        frame: Integer representing the frame number for the calculation

    Returns:
           fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)
    """

    # get pitch
    fig, ax = mviz.plot_frame(hometeam.loc[frame], awayteam.loc[frame])

    # Fill the area within the lines that we have drawn
    ax.fill(coordinates[hull.vertices, 0], coordinates[hull.vertices, 1], 'k', alpha=0.3)
    return fig, ax

def vis_team_centroid(hometeam, awayteam, x_centroid, y_centroid, frame=1, side='home'):
    """ Visualizes team centroid on the pitch

    Args:
        hometeam: row (i.e. instant) of the home team tracking data frame
        awayteam: row of the away team tracking data frame
        x_centroid: A float that represents the centroid with respect to x-axis
        y_centroid: A float that represents the centroid with respect to y-axis
        frame: Integer representing the frame number for the calculation
        side: boolean indicting if the team coordinates are from the home of away team

    Returns:
           fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)
    """
    # get basic pitch
    fig, ax = mviz.plot_frame(hometeam[frame], awayteam.loc[frame])

    # color
    if side == 'home':
        colorMarker = 'red'
    else:
        colorMarker = 'blue'

    # add centroid
    ax.plot(x_centroid, y_centroid, color=colorMarker, Marker='x', MarkerSize=10)
    return fig, ax

def vis_max_axis(hometeam, awayteam, coordinates, axis='y-axis', frame=1, side='home'):
    """ Visualizes line between two outest players on a axis

    Args:
        hometeam: row (i.e. instant) of the home team tracking data frame
        awayteam: row of the away team tracking data frame
        coordinates: List with the coordinates of the players that are used for calculation
        axis: either 'y-axis' or 'x-axis' indicating the axis to perform the action
        frame: Integer representing the frame number for the calculation
        side: boolean indicting if the team coordinates are from the home of away team

    Returns:
           fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)
    """
    # get basic pitch
    fig, ax = mviz.plot_frame(hometeam.loc[frame], awayteam.loc[frame])

    # Get coordinates of the extreme players
    x_max_player, y_max_player = fund.extrema_axis_player(coordinates, 'max', axis)
    x_min_player, y_min_player = fund.extrema_axis_player(coordinates, 'min', axis)

    # color
    if side == 'home':
        colorMarker = 'red'
    else:
        colorMarker = 'blue'

    # draw line
    ax.plot([x_max_player, x_min_player], [y_max_player, y_min_player], linestyle='solid', color=colorMarker, linewidth=8,
            alpha=0.3)
    return fig, ax
