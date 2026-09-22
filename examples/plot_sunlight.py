"""Plot image capture times colored by estimated sunlight.

This example intentionally depends on ``kwgis`` rather than making it a
KWCOCO dependency. Install the example dependencies separately when needed::

    pip install kwgis kwplot

Example:
    python examples/plot_sunlight.py path/to/data.kwcoco.json
"""

import argparse
import math
import warnings

import numpy as np
import pandas as pd


def estimate_image_sunlight(dset):
    """Estimate sunlight for images with ``geos_point`` and ``datetime``."""
    from kwgis.utils.util_sunlight import estimate_sunlight
    from kwutil.util_math import Rational

    sunlight_values = []
    for img in dset.images().objs:
        if 'geos_point' not in img:
            sunlight = math.nan
        else:
            geos_point = img['geos_point']
            if isinstance(geos_point, float) and math.isnan(geos_point):
                sunlight = math.nan
            elif not isinstance(geos_point, dict):
                warnings.warn(
                    f'Unknown geos_point format {geos_point!r} in {img!r}'
                )
                sunlight = math.nan
            else:
                coords = geos_point['coordinates']
                lon, lat = [Rational.coerce(x) for x in coords]
                datetime = img['datetime']
                sunlight = estimate_sunlight(lat, lon, datetime)
        sunlight_values.append(sunlight)
    return np.asarray(sunlight_values)


def build_plot_table(dset):
    """Build day-of-year / hour-of-day data with sunlight estimates."""
    import kwutil

    images = dset.images()
    datetimes = [
        a or b
        for a, b in zip(images.get('timestamp', None), images.get('datetime', None))
    ]
    datetimes = [kwutil.datetime.coerce(x) for x in datetimes]

    data = pd.DataFrame({'datetime': datetimes})
    data['day_of_year'] = [
        x.timetuple().tm_yday if not pd.isnull(x) else None for x in datetimes
    ]
    data['hour_of_day'] = [
        None
        if pd.isnull(x)
        else x.hour + x.minute / 60 + x.second / 3600
        for x in datetimes
    ]
    data['sunlight'] = estimate_image_sunlight(dset)
    return data


def plot_sunlight(dset):
    """Plot capture time colored by the estimated sunlight value."""
    import kwimage
    import kwplot

    sns = kwplot.autosns()
    data = build_plot_table(dset)
    palette = sns.color_palette('flare', n_colors=4, as_cmap=True).reversed()

    ax = sns.scatterplot(
        data=data,
        x='day_of_year',
        y='hour_of_day',
        hue='sunlight',
        palette=palette,
        legend=False,
    )
    kwplot.phantom_legend(
        {
            'Night': kwimage.Color.coerce(palette.colors[0]).as255(),
            'Day': kwimage.Color.coerce(palette.colors[-1]).as255(),
            'nan': 'blue',
        },
        mode='circle',
    )
    ax.set_title('Time Captured')
    ax.set_xlabel('Day of Year')
    ax.set_ylabel('Hour of Day')
    return ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('src', help='Path to a KWCOCO dataset')
    args = parser.parse_args()

    import kwcoco
    import matplotlib.pyplot as plt

    dset = kwcoco.CocoDataset.coerce(args.src)
    plot_sunlight(dset)
    plt.show()


if __name__ == '__main__':
    main()
