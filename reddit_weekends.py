import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import date


OUTPUT_TEMPLATE = (
    "Initial T-test p-value: {initial_ttest_p:.3g}\n"
    "Original data normality p-values: {initial_weekday_normality_p:.3g} {initial_weekend_normality_p:.3g}\n"
    "Original data equal-variance p-value: {initial_levene_p:.3g}\n"
    "Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n"
    "Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"
    "Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n"
    "Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n"
    "Weekly T-test p-value: {weekly_ttest_p:.3g}\n"
    "Mann-Whitney U-test p-value: {utest_p:.3g}"
)


def main():
    # -----------------------------------------------------------
    #                       load the data
    # -----------------------------------------------------------
    reddit_counts = sys.argv[1]
    counts = pd.read_json(sys.argv[1], lines=True)

    # -----------------------------------------------------------
    #                        perform ETL
    # -----------------------------------------------------------
    # 1) filter for subreddits in r/canada
    counts = counts[counts['subreddit'] == 'canada'].reset_index(drop=True)
    # 2) create a new column to extract the year from the data
    # -- https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.year.html
    counts['year'] = counts['date'].dt.year
    # 3) filter for years in 2012 and 2013
    counts = counts[counts['year'].isin([2012, 2013])].reset_index(drop=True)
    # 4) create a new column indicating if it's a weekday/weekend
    # -- https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.dt.weekday.html
    counts['weekend'] = counts['date'].dt.weekday.isin([5, 6]).astype(bool)
    # 5) created new dfs filtering for weekday/weekend
    weekday = counts[counts['weekend'] == False].reset_index(drop=True)
    weekend = counts[counts['weekend'] == True].reset_index(drop=True)

    # optional: plot the data
    # plt.plot(weekday['date'], weekday['comment_count'], 'b.', alpha=0.5)
    # plt.plot(weekend['date'], weekend['comment_count'], 'r.', alpha=0.5)
    # plt.xlabel('date')
    # plt.ylabel('count')
    # plt.title('comment counts per day')
    # plt.xticks(rotation=25)
    # plt.show()

    # -----------------------------------------------------------
    #                  default: trying a t-test
    # -----------------------------------------------------------
    wd = weekday['comment_count']
    we = weekend['comment_count']
    ttest = stats.ttest_ind(wd, we)
    # print(ttest)
    initial_ttest = ttest.pvalue

    # testing for t-test assumptions, normality and equivariance
    # -- https://ggbaker.ca/data-science/content/inferential.html
    # For these samples, we get p < 0.05, so we're going to reject: the distribution seems to not be normal.
    # print('\nbase tests')
    # print(stats.normaltest(wd).pvalue)
    # print(stats.normaltest(we).pvalue)
    # Like the normality test: H₀ is that the groups have equal variance. With small p, we reject that assumption.
    # print(stats.levene(wd, we).pvalue)
    initial_weekday_normality = stats.normaltest(wd).pvalue
    initial_weekend_normality = stats.normaltest(we).pvalue
    initial_levene = stats.levene(wd, we).pvalue
    
    # -----------------------------------------------------------
    #   fix 1: transforming data for better normality/variance
    # -----------------------------------------------------------
    
    # log transforms
    log_wd = np.log(wd)
    log_we = np.log(we)
    # testing for normality
    # print('\nlog tests')
    # print(stats.normaltest(log_wd).pvalue)
    # print(stats.normaltest(log_we).pvalue)
    # testing for equal variance
    # print(stats.levene(log_wd, log_we).pvalue)

    # exp transform 
    exp_wd = np.exp(wd)
    exp_we = np.exp(we)
    # testing for normality
    # print('\nexp tests')
    # print(stats.normaltest(exp_wd).pvalue)
    # print(stats.normaltest(exp_we).pvalue)
    # testing for equal variance
    # print(stats.levene(exp_wd, exp_we).pvalue)

    # sqrt transform
    sqrt_wd = np.sqrt(wd)
    sqrt_we = np.sqrt(we)
    # testing for normality
    # print('\nsqrt tests')
    # print(stats.normaltest(sqrt_wd).pvalue)
    # print(stats.normaltest(sqrt_we).pvalue)
    # testing for equal variance
    # print(stats.levene(sqrt_wd, sqrt_we).pvalue)

    # square transform
    sqr_wd = wd**2
    sqr_we = we**2
    # testing for normality
    # print('\nsquare tests')
    # print(stats.normaltest(sqr_wd).pvalue)
    # print(stats.normaltest(sqr_we).pvalue)
    # testing for equal variance
    # print(stats.levene(sqr_wd, sqr_we).pvalue)

    # choosing the best results
    transformed_weekday_normality = stats.normaltest(sqrt_wd).pvalue
    transformed_weekend_normality = stats.normaltest(sqrt_we).pvalue
    transformed_levene = stats.levene(sqrt_wd, sqrt_we).pvalue

    # -----------------------------------------------------------
    #         fix 2: applying the central limit theorem
    # -----------------------------------------------------------
    
    # -- https://ggbaker.ca/data-science/content/stats-tests.html
    # -- https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.isocalendar.html
    # changing extracting week data into its own column
    wdcounts = counts
    wdcounts['week'] = wdcounts['date'].dt.isocalendar().week
    wdcounts = wdcounts[wdcounts['weekend'] == False].reset_index(drop=True)
    # kept getting an issue when aggregating, needed to drop object columns
    wdcounts = wdcounts.drop(columns=['subreddit'])

    wecounts = counts
    wecounts['week'] = wecounts['date'].dt.isocalendar().week
    wecounts = wecounts[wecounts['weekend'] == True].reset_index(drop=True)
    # kept getting an issue when aggregating, needed to drop object columns
    wecounts = wecounts.drop(columns=['subreddit'])

    # create aggregated data sets to apply central limit theorem
    aggwd = wdcounts.groupby(by=['year', 'week']).mean()
    aggwe = wecounts.groupby(by=['year', 'week']).mean()

    # testing for normality and equivariance
    # print('\nfix 2 applied')
    # print(stats.normaltest(aggwd['comment_count']).pvalue)
    # print(stats.normaltest(aggwe['comment_count']).pvalue)
    # print(stats.levene(aggwd['comment_count'], aggwe['comment_count']).pvalue)
    weekly_weekday_normality = stats.normaltest(aggwd['comment_count']).pvalue
    weekly_weekend_normality = stats.normaltest(aggwe['comment_count']).pvalue
    weekly_levene = stats.levene(aggwd['comment_count'], aggwe['comment_count']).pvalue
    
    # performing a t-test
    fix2_ttest = stats.ttest_ind(aggwd['comment_count'], aggwe['comment_count'])
    weekly_ttest = fix2_ttest.pvalue
    # print(weekly_ttest)

    # -----------------------------------------------------------
    #           fix 3: applying a mann-whitney u-test
    # -----------------------------------------------------------
    # -- https://ggbaker.ca/data-science/content/stats-tests.html
    # -- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
    # print('\nfix 3 applied')
    # print(stats.mannwhitneyu(wd, we).pvalue)
    utest = stats.mannwhitneyu(wd, we).pvalue

    # -----------------------------------------------------------
    #                      print the output
    # -----------------------------------------------------------
    print('\n---------------- output results ----------------')
    print(OUTPUT_TEMPLATE.format(
        initial_ttest_p=initial_ttest,
        initial_weekday_normality_p=initial_weekday_normality,
        initial_weekend_normality_p=initial_weekend_normality,
        initial_levene_p=initial_levene,
        transformed_weekday_normality_p=transformed_weekday_normality,
        transformed_weekend_normality_p=transformed_weekend_normality,
        transformed_levene_p=transformed_levene,
        weekly_weekday_normality_p=weekly_weekday_normality,
        weekly_weekend_normality_p=weekly_weekend_normality,
        weekly_levene_p=weekly_levene,
        weekly_ttest_p=weekly_ttest,
        utest_p=utest,
    ))

if __name__ == '__main__':
    main()