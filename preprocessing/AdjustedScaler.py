import numpy as np
from pandas import DataFrame
import math


class AdjustedScaler():
    """Scale features using statistics that are robust to outliers.

    This scaler scales the data to a valid range to eliminate potential outliers.
    The range of significant values is calculated using the medcouple function (MC)
    according to the principle proposed by M. Huberta and E. Vandervierenb in "An
    adjusted boxplot for skewed distributions" Computational Statistics & Data Analysis,
    vol. 52, pp. 5186-5201, August 2008.

    Parameters
        ----------
        with_centering : boolean, True by default
            If True, center the data before scaling

        paired: list, tuple, False by default
            Paired features names

    .. versionadded:: 0.2
    """

    def __init__(self, with_centering=True, paired=False):
        self.scaling_parameters = {}
        self.with_centering = with_centering
        self.paired_columns = paired

    @staticmethod
    def _check_data(x):
        if not isinstance(x, DataFrame):
            raise ValueError('Invalid type of "x" parameter')

    def fit(self, x):
        """Compute the median and adjusted interval to be used for scaling.

        Parameters
        ----------
        x : DataFrame
            Pandas DataFrame Object whose data needs to be converted.
        """

        self._check_data(x)

        for column_name in x.columns:
            median = x[column_name].median()
            quantile_1 = x[column_name].quantile(q=0.25)
            quantile_3 = x[column_name].quantile(q=0.75)
            iqr = quantile_3 - quantile_1
            mc = self.medcouple(y=x[column_name], axis=0)

            if mc < 0.0:
                adjusted_interval = (
                    quantile_1 - 1.5 * math.exp(-3 * mc) * iqr, quantile_3 + 1.5 * math.exp(4 * mc) * iqr)
            else:
                adjusted_interval = (
                    quantile_1 - 1.5 * math.exp(-4 * mc) * iqr, quantile_3 + 1.5 * math.exp(3 * mc) * iqr)

            existing_border_left = x[x[column_name] >= adjusted_interval[0]][column_name].min()
            existing_border_right = x[x[column_name] <= adjusted_interval[1]][column_name].max()

            if existing_border_left == existing_border_right:
                existing_border_right = x[column_name].quantile(q=0.95)
                existing_border_left = x[column_name].quantile(q=0.05)
            if existing_border_left == existing_border_right:
                existing_border_right = x[column_name].max()
                existing_border_left = x[column_name].min()

            adjusted_scale_value = existing_border_right - existing_border_left
            if adjusted_scale_value == 0.0:
                adjusted_scale_value = 1

            if self.with_centering:
                delta = median
            else:
                delta = existing_border_left

            self.scaling_parameters[column_name] = {
                'delta': delta,
                'scale_interval_border_left': existing_border_left,
                'scale_interval_border_right': existing_border_right,
                'scale_interval': adjusted_scale_value
            }

        self._check_paired_features(self.paired_columns)

        return self

    def _check_paired_features(self, paired_columns):
        """Paired feature processing.

        Parameters
        ----------
        paired_columns : tuple, list
            All paired features names.
        """

        if isinstance(paired_columns, tuple):
            self._process_paired_features(paired_columns)
        elif isinstance(paired_columns, list):
            for columns in paired_columns:
                self._check_paired_features(columns)

    def _process_paired_features(self, paired_columns):
        """Calculates scaling parameters for paired features.

        Parameters
        ----------
        paired_columns : tuple
            Paired feature names.
        """

        first_column_name = paired_columns[0]
        max_scale_interval = self.scaling_parameters[first_column_name]['scale_interval']
        for i in range(len(paired_columns)):
            column_name = paired_columns[i]
            scale_interval = self.scaling_parameters[column_name]['scale_interval']
            if max_scale_interval < scale_interval:
                max_scale_interval = scale_interval

        # updating all scale interval values for paired columns
        for column_name in paired_columns:
            self.scaling_parameters[column_name]['scale_interval'] = max_scale_interval

    def transform(self, x):
        """Center and scale the data.

        Parameters
        ----------
        x : DataFrame
            Pandas DataFrame Object whose data needs to be converted.

        Returns
        -------
        x_scaled : DataFrame
            Pandas DataFrame Object with converted data
        """

        self._check_data(x)

        x_scaled = x.copy()

        for column_name in x_scaled.columns:
            delta = self.scaling_parameters[column_name]['delta']
            scale_interval = self.scaling_parameters[column_name]['scale_interval']
            x_scaled[column_name] = (x_scaled[column_name] - delta) / scale_interval

        return x_scaled

    def inverse_transform(self, x_scaled):
        """Scale back the data to the original representation

        Parameters
        ----------
        x_scaled : DataFrame
            Pandas DataFrame Object whose data needs to be restored.

        Returns
        -------
        x : DataFrame
            Pandas DataFrame Object with restored data
        """

        self._check_data(x_scaled)

        x = x_scaled.copy()

        for column_name in x.columns:
            delta = self.scaling_parameters[column_name]['delta']
            scale_interval = self.scaling_parameters[column_name]['scale_interval']
            x[column_name] = x[column_name] * scale_interval + delta

        return x

    def _medcouple_1d(self, y):
        """
        Calculates the medcouple robust measure of skew.

        Parameters
        ----------
        y : array_like, 1-d
            Data to compute use in the estimator.

        Returns
        -------
        mc : float
            The medcouple statistic

        Notes
        -----
        The current algorithm requires a O(N**2) memory allocations, and so may
        not work for very large arrays (N>10000).

        .. [*] M. Huberta and E. Vandervierenb, "An adjusted boxplot for skewed
           distributions" Computational Statistics & Data Analysis, vol. 52, pp.
           5186-5201, August 2008.
        """

        # Parameter changes the algorithm to the slower for large n

        y = np.squeeze(np.asarray(y))
        if y.ndim != 1:
            raise ValueError("y must be squeezable to a 1-d array")

        y = np.sort(y)

        n = y.shape[0]
        if n % 2 == 0:
            mf = (y[n // 2 - 1] + y[n // 2]) / 2
        else:
            mf = y[(n - 1) // 2]

        z = y - mf
        lower = z[z <= 0.0]
        upper = z[z >= 0.0]
        upper = upper[:, None]
        standardization = upper - lower
        is_zero = np.logical_and(lower == 0.0, upper == 0.0)
        standardization[is_zero] = np.inf
        spread = upper + lower
        h = spread / standardization
        # GH5395
        num_ties = np.sum(lower == 0.0)
        if num_ties:
            # Replacements has -1 above the anti-diagonal, 0 on the anti-diagonal,
            # and 1 below the anti-diagonal
            replacements = np.ones((num_ties, num_ties)) - np.eye(num_ties)
            replacements -= 2 * np.triu(replacements)
            # Convert diagonal to anti-diagonal
            replacements = np.fliplr(replacements)
            # Always replace upper right block
            h[:num_ties, -num_ties:] = replacements

        return np.median(h)

    def medcouple(self, y, axis=0):
        """
        Calculate the medcouple robust measure of skew.

        Parameters
        ----------
        y : array_like
            Data to compute use in the estimator.
        axis : {int, None}
            Axis along which the medcouple statistic is computed.  If `None`, the
            entire array is used.

        Returns
        -------
        mc : ndarray
            The medcouple statistic with the same shape as `y`, with the specified
            axis removed.

        Notes
        -----
        The current algorithm requires a O(N**2) memory allocations, and so may
        not work for very large arrays (N>10000).

        .. [*] M. Huberta and E. Vandervierenb, "An adjusted boxplot for skewed
           distributions" Computational Statistics & Data Analysis, vol. 52, pp.
           5186-5201, August 2008.
        """
        y = np.asarray(y, dtype=np.double)  # GH 4243
        if axis is None:
            return self._medcouple_1d(y.ravel())

        return np.apply_along_axis(self._medcouple_1d, axis, y)
