"""
Created by Created by Hadi Afshar 
2019-08-19
"""

import numpy as np
import pandas

def mice_data(file_path, x_columns_aliases, y_column, arrival_waves_to_choose=None):
    """
    :param file_path: path of the CSV file
    :param x_columns_aliases: aliases to refer to X_columns
    :param y_column: output
    :param arrival_waves_to_choose: None, or a list based on which 1 or more batches are selected
    :return:
    """
    lifespan = pandas.read_csv(file_path)

    if arrival_waves_to_choose:
        lifespan = lifespan[lifespan['Arrival.wave'].isin(arrival_waves_to_choose)]
    print('lifespan.shape: ', lifespan.shape)


    X_cols = [xa[0] for xa in x_columns_aliases]
    X_names = [xa[1] for xa in x_columns_aliases]

    XandYandFolds = lifespan[X_cols + [y_column, 'fold']]
    del lifespan
    XandYandFolds = XandYandFolds.replace('', np.nan)
    XandYandFolds = XandYandFolds.dropna()

    X = XandYandFolds[X_cols]
    Y = XandYandFolds[y_column]
    folds = XandYandFolds['fold'].to_numpy()

    X = X.to_numpy()
    Y = Y.to_numpy()

    Y = np.expand_dims(Y, axis=1)  # to convert the 1D [y_0, y_1, ...] to 2D [[y_0], [y_1], ...]

    # To Add not Linearty for test:
    Prt = XandYandFolds['P eaten (kJ/mse/cage/d)'].to_numpy()
    Crb = XandYandFolds['C eaten (kJ/mse/cage/d)'].to_numpy()
    Fat = XandYandFolds['F eaten (kJ/mse/cage/d)'].to_numpy()

    Prt = np.expand_dims(Prt, axis=1)
    Crb = np.expand_dims(Crb, axis=1)
    Fat = np.expand_dims(Fat, axis=1)

    # X = np.append(X, Prt * Crb, axis=1)
    # X_names.extend(['PxC'])
    # X = np.append(X, Prt * Fat, axis=1)
    # X_names.extend(['PxF'])
    # X = np.append(X, Crb * Fat, axis=1)
    # X_names.extend(['CxF'])
    # X = np.append(X, Prt * Crb * Fat, axis=1)
    # X_names.extend(['PxCxF'])

    # X = np.append(X, Prt / (Prt + Crb + Fat), axis=1)
    # X_names.extend(['P/(P+C+F)'])
    # X = np.append(X, Crb / (Prt + Crb + Fat), axis=1)
    # X_names.extend(['C/(P+C+F)'])
    # X = np.append(X, Fat / (Prt + Crb + Fat), axis=1)
    # X_names.extend(['F/(P+C+F)'])
    X = np.append(X, Prt / Crb, axis=1)
    X_names.extend(['P/C'])

    assert len(X_names) == X.shape[1]

    ################

    if False:
        plt.subplot(221)
        plt.plot(Prt[:, 0]/Crb[:, 0], Y[:, 0], 'r.')
        plt.xlabel('P/C')
        plt.ylabel('Life')

        plt.subplot(222)
        plt.plot(Crb[:, 0], Y[:, 0], 'r.')
        plt.xlabel('C')
        plt.ylabel('Life')

        plt.subplot(223)
        plt.plot(Prt[:, 0], Crb[:, 0], 'r.')
        plt.xlabel('P')
        plt.ylabel('C')

        plt.subplot(224)
        plt.plot(Prt[:, 0], Fat[:, 0], 'r.')
        plt.xlabel('P')
        plt.ylabel('F')

        plt.show()

    ################
    return X, Y, X_names, folds


def main():
    mX, vY, vX_names, folds = mice_data(
        file_path='../lifespan-merged-folded.csv',
        x_columns_aliases=[
            ('Dry weight food eaten (g/mouse/cage/d)', 'Dry'),
            ('Cellulose intake (g/d)', 'Cel'),
            ('P eaten (kJ/mse/cage/d)', 'P'),
            ('C eaten (kJ/mse/cage/d)', 'C'),
            ('F eaten (kJ/mse/cage/d)', 'F'),
            ('Energy intake (kJ/mse/cage/d)', 'En')
        ],
        y_column='age at death (w)',
        arrival_waves_to_choose=['First', 'Second', 'Third']
    )
    print('vX_names: ', vX_names)
    print('mX.shape: ', mX.shape)
    print('vY.shape', vY.shape)


if __name__ == "__main__":
    main()