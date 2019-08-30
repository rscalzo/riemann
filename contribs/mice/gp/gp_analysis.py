"""
Created by Hadi Afshar
August 2019
"""
from contribs.mice.gp.gp_utils import *


def fetch_data(test_folds):
    # 1. fetch data:
    data_gen_strategy = 'fake_gp'
    print('Data generation strategy:\t', data_gen_strategy)
    # data_gen_strategy = 'fake_linear'
    # data_gen_strategy = 'mice'

    true_theta = None
    if data_gen_strategy == 'fake_gp':
        n_train = 100  # num data points
        q = 1  # number of features
        lower_bound = -1
        upper_bound = 1

        true_theta = Theta(sigma2_f=2.5, l=0.8, sigma2_e=0.005)
        mX_train, vy_train = create_fake_data(n=n_train, q=q, theta=true_theta,
                                              lower_bound=lower_bound, upper_bound=upper_bound)
        num_test_points = 100
        mX_test = np.random.uniform(low=lower_bound, high=upper_bound, size=(num_test_points, q))
        # sort them (for plotting):
        # mX_test = np.sort(mX_test, axis=0) #
        vy_test = None
        x_names = None

    elif data_gen_strategy == 'fake_linear':
        n_train = 50  # num data points
        q = 1  # number of features
        lower_bound = -1
        upper_bound = 1
        mX_train, vy_train = create_fake_data_linear(n=n_train, q=q, theta=Theta(sigma2_f=1, l=2, sigma2_e=0.05),
                                                     lower_bound=lower_bound, upper_bound=upper_bound)
        num_test_points = 100
        mX_test = np.random.uniform(low=lower_bound, high=upper_bound, size=(num_test_points, q))
        # sort them (for plotting):
        # mX_test = np.sort(mX_test, axis=0) #todo this is wrong for D >1 (??)
        vy_test = None
        x_names = None

    elif data_gen_strategy == 'mice':
        mX_all, vy_all, x_names, folds = mice_data.mice_data(
            file_path='../lifespan-merged-folded.csv',
            x_columns_aliases=[
                # ('Dry weight food eaten (g/mouse/cage/d)', 'Dry'),
                # ('Cellulose intake (g/d)', 'Cel'),
                ('P eaten (kJ/mse/cage/d)', 'P'),
                ('C eaten (kJ/mse/cage/d)', 'C'),
                ('F eaten (kJ/mse/cage/d)', 'F'),
                # ('Energy intake (kJ/mse/cage/d)', 'En')
            ],
            y_column='age at death (w)',
            arrival_waves_to_choose=['First', 'Second', 'Third']
        )
        n_all = mX_all.shape[0]
        # test_folds = [0]  # todo
        test_indices = [i for i in range(n_all) if folds[i] in test_folds]
        train_indices = [i for i in range(n_all) if folds[i] not in test_folds]

        mX_train = mX_all[train_indices]
        vy_train = vy_all[train_indices]
        mX_test = mX_all[test_indices]
        vy_test = vy_all[test_indices]

        assert mX_train.shape[0] + mX_test.shape[0] == n_all

        # standardization:
        #### Some Stats:
        print("##### SOME STATS #######")
        print('X_train.shape: {:10s} \t\t X_test.shape: {:10s}'.format(str(mX_train.shape), str(mX_test.shape)))
        x_train_min_before_standard = np.amin(mX_train, axis=0)
        x_train_max_before_standard = np.amax(mX_train, axis=0)
        x_test_min_before_standard = np.amin(mX_test, axis=0)
        x_test_max_before_standard = np.amax(mX_test, axis=0)

        y_train_min_before_standard = np.amin(vy_train, axis=0)
        y_train_max_before_standard = np.amax(vy_train, axis=0)
        y_test_min_before_standard = np.amin(vy_test, axis=0)
        y_test_max_before_standard = np.amax(vy_test, axis=0)

        # print('Y_train range: [{:8.3f}, {:8.3f}]'.format(np.amin(vy_train), np.amax(vy_train)))

        # Scaling should be performed based on training data only:
        scale_data = True
        if scale_data:
            # Note: here we do not have any 1 vector in the feature matrix
            x_scaler = preprocessing.StandardScaler()
            mX_train = x_scaler.fit_transform(mX_train)
            mX_test = x_scaler.transform(mX_test)
            y_scaler = preprocessing.StandardScaler()
            vy_train = y_scaler.fit_transform(vy_train)
            vy_test = y_scaler.transform(vy_test)

        x_train_min_after_standard = np.amin(mX_train, axis=0)
        x_train_max_after_standard = np.amax(mX_train, axis=0)
        x_test_min_after_standard = np.amin(mX_test, axis=0)
        x_test_max_after_standard = np.amax(mX_test, axis=0)

        num_features = mX_train.shape[1]
        for i in range(num_features):
            print('Column {:10s}\t'.format(x_names[i]),
                  'Train range: [{:8.3f}, {:8.3f}] =>'.format(x_train_min_before_standard[i],
                                                              x_train_max_before_standard[i]),
                  '[{:8.3f}, {:8.3f}]\t\t'.format(x_train_min_after_standard[i], x_train_max_after_standard[i]),
                  'Test range: [{:8.3f}, {:8.3f}] =>'.format(x_test_min_before_standard[i],
                                                             x_test_max_before_standard[i]),
                  '[{:8.3f}, {:8.3f}]'.format(x_test_min_after_standard[i], x_test_max_after_standard[i]))

        y_train_min_after_standard = np.amin(vy_train, axis=0)
        y_train_max_after_standard = np.amax(vy_train, axis=0)
        y_test_min_after_standard = np.amin(vy_test, axis=0)
        y_test_max_after_standard = np.amax(vy_test, axis=0)

        print('Output \t\t\t\tTrain range: [{:8.3f}, {:8.3f}] =>'.format(y_train_min_before_standard[0],
                                                                         y_train_max_before_standard[0]),
              '[{:8.3f}, {:8.3f}]\t\t'.format(y_train_min_after_standard[0], y_train_max_after_standard[0]),
              'Test range: [{:8.3f}, {:8.3f}] =>'.format(y_test_min_before_standard[0],
                                                         y_test_max_before_standard[0]),
              '[{:8.3f}, {:8.3f}]'.format(y_test_min_after_standard[0], y_test_max_after_standard[0]))

        # note: vy_train (and test) are [n_train x 1] (and n_test x1] matrices.
        # We should reshape then to vectors:
        vy_train = vy_train.reshape(vy_train.shape[0])
        vy_test = vy_test.reshape(vy_test.shape[0])
        assert len(vy_train) + len(vy_test) == n_all

    else:
        raise Exception("Unknown data generation strategy: ", data_gen_strategy)
    return mX_train, vy_train, mX_test, vy_test, x_names, true_theta


def main():
    mX_train, vy_train, mX_test, vy_test, x_names, true_theta = fetch_data(test_folds=[0])

    # in a 2D plot we only plot y versus one dimension (or feature of X):
    dim_to_plot = 0  # todo
    # sort test data for plotting:
    mX_test, vy_test = sort_based_on_dim(mX=mX_test, dim_to_sort_on=dim_to_plot, vy=vy_test)

    # print('mX', mX_train)
    # print('vY', vy_train)

    should_plot_training_data = False
    if should_plot_training_data:
        do_plot('Training Data', mX_train[:, 0], vy_train)

    # Plotting data / samples from a posterior of a GP with those same parameters.
    plot_gp_posterior_with_true_params = False
    if plot_gp_posterior_with_true_params:
        f_star_mean, f_star_cov = calc_f_star_mean_cov(mX_test=mX_test, vy=vy_train, mX=mX_train, theta=true_theta)
        plt.figure()
        plt.plot(mX_train[:, 0], vy_train, 'bo')
        plt.plot(mX_test[:, 0], f_star_mean, color='k', alpha=0.7, lw=3)
        f_star_std = np.power(f_star_cov.diagonal(), 0.5)
        plt.fill_between(x=mX_test[:, 0], y1=f_star_mean - 2 * f_star_std, y2=f_star_mean + 2 * f_star_std, color='k',
                         alpha=0.2)
        plt.show()

    # initialize:    true_theta = Theta(sigma2_f=2.5, l=0.8, sigma2_e=0.005)
    theta_init = Theta(sigma2_f=1, l=1, sigma2_e=0.5)

    # theta_curr = Theta(sigma2_f=1, l=1, sigma2_e=0.4)
    theta_curr = theta_init.copy()

    num_f_samples_to_plot = 20
    f_samples_to_plot = []

    should_show_histograms = True
    if should_show_histograms:
        all_samples_so_far = []

    num_samples = 10000

    sum_sampled_thetas = Theta(0, 0, 0)
    sum_sampled_fs = np.zeros(len(mX_test))  # to compute E[f* | .]
    sum_sampled_f2s = np.zeros(len(mX_test))  # to compute E[f*^2 | .] and from there its variance
    for sample_count in tqdm(range(1, num_samples + 1)):
        theta_prop = theta_curr.propose(eps_f=0.001, eps_l=0.001, eps_e=0.001)

        theta_prop.sigma2_f = true_theta.sigma2_f; theta_init.sigma2_f = true_theta.sigma2_f  # todo just test
        # theta_prop.l = true_theta.l    # todo just test
        theta_prop.sigma2_e = true_theta.sigma2_e; theta_init.sigma2_e = true_theta.sigma2_e  # just for test

        curr_likelihood = pdf_y_given_params(vy_train, theta_curr, mX_train)
        prop_likelihood = pdf_y_given_params(vy_train, theta_prop, mX_train)
        if curr_likelihood == 0:
            print("The probability of the current likelihood == 0!!!")
            alpha = 1
        else:
            alpha = min(1, prop_likelihood / curr_likelihood)

        if np.random.uniform(0, 1) < alpha:
            theta_curr = theta_prop

        f_star_mean, f_star_cov = calc_f_star_mean_cov(mX_test=mX_test, vy=vy_train, mX=mX_train, theta=theta_curr)
        f_sample = np.random.multivariate_normal(mean=f_star_mean, cov=f_star_cov)
        sum_sampled_fs += f_sample
        sum_sampled_f2s += f_sample ** 2
        if sample_count % int(num_samples / num_f_samples_to_plot) == 0:
            f_samples_to_plot.append(f_sample)

        sum_sampled_thetas.addTo(theta_curr)

        if should_show_histograms:
            all_samples_so_far.append(theta_curr)

    expected_theta = sum_sampled_thetas.mult(1 / num_samples)
    print('expected_theta:', expected_theta)

    # expect_f_star_mean, expect_f_star_cov = calc_f_star_mean_cov(mX_test=mX_test, vy=vy_train, mX=mX_train,
    #                                                              theta=expected_theta)
    # plt.plot(mX_test[:, dim_to_plot], expect_f_star_mean, color='k', alpha=0.6)
    # expect_f_star_std = np.power(expect_f_star_cov.diagonal(), 0.5)  # todo is this right?
    # plt.fill_between(x=mX_test[:, dim_to_plot], y1=expect_f_star_mean - 2 * expect_f_star_std,
    #                  y2=expect_f_star_mean + 2 * expect_f_star_std, color='k', alpha=0.1)
    f_sample_mean = sum_sampled_fs / num_samples
    f_sample_std = ((sum_sampled_f2s / num_samples) - (f_sample_mean ** 2)) ** 0.5  # root of E[f^2] - E[f]^2

    if True:
        plt.figure()
        plt.plot(mX_test[:, dim_to_plot], f_sample_mean, color='k', alpha=0.6)
        plt.fill_between(x=mX_test[:, dim_to_plot], y1=f_sample_mean - 2 * f_sample_std,
                         y2=f_sample_mean + 2 * f_sample_std, color='k', alpha=0.1)

        for f in f_samples_to_plot:
            plt.plot(mX_test[:, dim_to_plot], f, 'r', alpha=0.1)
        plt.plot(mX_train[:, dim_to_plot], vy_train, 'bo', alpha=0.9)
        plt.show()

    if should_show_histograms:
        fig = plt.figure()
        figtext = 'data={d} points; iter={i}; init param $\Theta_0= <\sigma^2_f: {sf:4.2f}, l: {l:4.2f}, \sigma^2_e: {se:4.3f}>$'.format(
            d=mX_train.shape[0], i=num_samples, sf=theta_init.sigma2_f, l=theta_init.l, se=theta_init.sigma2_e)

        if true_theta:
            figtext += '; true param $\Theta_T= <\sigma^2_f: {sf:4.2f}, l: {l:4.2f}, \sigma^2_e: {se:4.3f}>$'.format(
                sf=true_theta.sigma2_f, l=true_theta.l, se=true_theta.sigma2_e)
        fig.suptitle(figtext)

        plt.subplot(2, 3, 1)

        _ = plt.hist([theta.sigma2_f for theta in all_samples_so_far], bins='auto')
        if true_theta:
            plt.axvline(true_theta.sigma2_f, color='r', linestyle='dashed', linewidth=3)
        plt.axvline(expected_theta.sigma2_f, color='k', linestyle='dashed', linewidth=1)
        plt.title('Histogram of sampled $\sigma^2_f$')
        # plt.show()

        plt.subplot(2, 3, 2)
        _ = plt.hist([theta.l for theta in all_samples_so_far], bins='auto')
        if true_theta:
            plt.axvline(true_theta.l, color='r', linestyle='dashed', linewidth=3)
        plt.axvline(expected_theta.l, color='k', linestyle='dashed', linewidth=1)
        plt.title('Histogram of sampled $l$')

        plt.subplot(2, 3, 3)
        _ = plt.hist([theta.sigma2_e for theta in all_samples_so_far], bins='auto')
        if true_theta:
            plt.axvline(true_theta.sigma2_e, color='r', linestyle='dashed', linewidth=3)
        plt.axvline(expected_theta.sigma2_e, color='k', linestyle='dashed', linewidth=1)
        plt.title('Histogram of sampled $\sigma^2_e$')

        plt.subplot(2, 3, 4)
        if true_theta:
            plt.axhline(true_theta.sigma2_f, color='r', linestyle='dashed', linewidth=3)
        plt.title('Trace plot')
        plt.ylabel('$(\sigma_f^2)^{[t]}$')
        plt.xlabel('iteration (t)')
        plt.plot(range(1, num_samples + 1), [theta.sigma2_f for theta in all_samples_so_far])

        plt.subplot(2, 3, 5)
        if true_theta:
            plt.axhline(true_theta.l, color='r', linestyle='dashed', linewidth=3)
        plt.title('Trace plot')
        plt.ylabel('$l^{[t]}$')
        plt.xlabel('iteration (t)')
        plt.plot(range(1, num_samples + 1), [theta.l for theta in all_samples_so_far])

        plt.subplot(2, 3, 6)
        if true_theta:
            plt.axhline(true_theta.sigma2_e, color='r', linestyle='dashed', linewidth=3)
        plt.title('Trace plot')
        plt.ylabel('$(\sigma^2_e)^{[t]}$')
        plt.xlabel('iteration (t)')
        plt.plot(range(1, num_samples + 1), [theta.sigma2_e for theta in all_samples_so_far])

        plt.show()


if __name__ == "__main__":
    main()
