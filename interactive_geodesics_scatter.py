'''
Imports for the Metrics/Helper functions
'''
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from einsteinpy.metric import Schwarzschild

'''
Imports for Plotly
'''
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import  init_notebook_mode, plot, iplot
import plotly.io as pio  # To save the Plot as an Image


class InteractiveScatterGeodesicPlotter:
    """
    Class for plotting interactive plotly plots.
    """

    def __init__(
        self, mass, time=0 * u.s, attractor_color="black", cmap_color="Orange",connected=True
    ):
        """
        Parameters
        ----------
        attractor_color : string, optional
            Color which is used to denote the attractor. Defaults to black.
        cmap_color : string, optional
            Color used in function plot.
        """
        self.mass = mass
        self.time = time
        self.data = []
        self.trace = None
        self.fig = None
        #init_notebook_mode(connected=False)
        # self.steps = 
        self.connected = connected
        self._attractor_present = False
        self.attractor_color = attractor_color
        self.layout = dict(
            title = 'Interactive Plot',
            yaxis = dict(zeroline = False),
            xaxis = dict(zeroline = False)
             )
        self.cmap_color = cmap_color

    def _plot_attractor(self):

        self._attractor_present = True
        self.data.append(go.Scatter(
                                x = [0],
                                y = [0],
                                mode = 'markers',
                                name = 'attractor',
                                marker = dict(
                                    size = 5,
                                    color = self.attractor_color,
                                    line = dict(
                                        width = 2,
                                    )
                                )
                            ))

    def plot(self, coords, end_lambda=10, step_size=1e-3):
        """
        Parameters
        ----------
        coords : ~einsteinpy.coordinates.velocity.SphericalDifferential
            Position and velocity components of particle in Spherical Coordinates.
        end_lambda : float, optional
            Lambda where iteartions will stop.
        step_size : float, optional
            Step size for the ODE.
        """

        swc = Schwarzschild.from_spherical(coords, self.mass, self.time)

        vals = swc.calculate_trajectory(
            end_lambda=end_lambda, OdeMethodKwargs={"stepsize": step_size}
        )[1]

        time = vals[:, 0]
        r = vals[:, 1]
        # Currently not being used (might be useful in future)
        # theta = vals[:, 2]
        phi = vals[:, 3]

        pos_x = r * np.cos(phi)
        pos_y = r * np.sin(phi)

        self.data.append(
            go.Scatter(
                                x = pos_x,
                                y = pos_y,
                                mode = 'markers',
                                name = 'plot_points',
                                marker = dict(
                                    size = 5,
                                    color = self.cmap_color,   
                                    line = dict(
                                        width = 2,
                                    )
                                )
                            )
        )

        init_notebook_mode(connected=self.connected)


        self.fig = dict(data=self.data, layout=self.layout)
        # plt.scatter(pos_x, pos_y, s=1, c=time, cmap=self.cmap_color)

        if not self._attractor_present:
            self._plot_attractor()

    # TODO: WIP
    # SUGGESTION: Have another function to do the preprocessing and then update the State(self.) of the Object so that both animate() and plot()  have access to pos_x/pos_y etc.
    def animate(self, coords, end_lambda=10, step_size=1e-3, interval=50):
        """
        Function to generate animated plots of geodesics.
        Parameters
        ----------
        coords : ~einsteinpy.coordinates.velocity.SphericalDifferential
            Position and velocity components of particle in Spherical Coordinates.
        end_lambda : float, optional
            Lambda where iteartions will stop.
        step_size : float, optional
            Step size for the ODE.
        interval : int, optional
            Control the time between frames. Add time in milliseconds.
        """

        swc = Schwarzschild.from_spherical(coords, self.mass, self.time)

        vals = swc.calculate_trajectory(
            end_lambda=end_lambda, OdeMethodKwargs={"stepsize": step_size}
        )[1]

        time = vals[:, 0]
        r = vals[:, 1]
        # Currently not being used (might be useful in future)
        # theta = vals[:, 2]
        phi = vals[:, 3]

        pos_x = r * np.cos(phi)
        pos_y = r * np.sin(phi)
        x_max, x_min = max(pos_x), min(pos_x)
        y_max, y_min = max(pos_y), min(pos_y)
        margin_x = (x_max - x_min) * 0.1
        margin_y = (y_max - y_min) * 0.1


        self.data = [dict(
                            visible = False,
                            line=dict(color='#00CED1', width=6),
                            name = '',
                            x = pos_x[:i],
                            y = pos_y[:i]) for i in range(0,len(pos_x),100)]

        steps = []
        for i in range(len(self.data)):
            step = dict(
                method = 'restyle',  
                args = ['visible', [False] * len(self.data)],
            )
            step['args'][1][i] = True # Toggle i'th trace to "visible"
            steps.append(step)

        self.layout=dict(sliders=[dict
                                (
                                    active = 100,
                                    currentvalue = {"prefix": "Frequency: "},
                                    pad = {"t": 50},
                                    steps = steps
                                )],
                            title ='[EINSTEINPY] Plotly Animation', hovermode='closest',
                            xaxis =dict(
                                tickmode ='linear',
                                ticks ='outside',
                                tick0 = int(x_min - margin_x),
                                dtick = int((x_max + margin_x -x_min - margin_x))/10,
                                ticklen = 8,
                                tickwidth = 4,
                                tickcolor ='#000'
                            ),
                            yaxis=dict(
                                tickmode='linear',
                                ticks='outside',
                                tick0=int(y_min - margin_y),
                                dtick=int((y_max + margin_y -y_min - margin_y)/10),
                                ticklen=8,
                                tickwidth=4,
                                tickcolor='#000'
                            ),
                            updatemenus= [{'type': 'buttons',
                                        'buttons': [{'label': 'Play',
                                                        'method': 'animate',
                                                        'args': [None]}]}])
        self.fig= dict(data=self.data, layout=self.layout)                                                
        # def _update(frame):
        #     pic.set_offsets(np.vstack((pos_x[: frame + 1], pos_y[: frame + 1])).T)
        #     pic.set_array(time[: frame + 1])
        #     return (pic,)


    def show(self):
        iplot(self.fig, filename='InteractiveScatterGeodesicPlotter')

# TODO: WIP
    def save(self, name="interactive_scatter_geodesic.jpeg"):
        pio.write_image(self.fig, name)
