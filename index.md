---
title: "The Cignetti and The Noise: A machine learning model for predicting Power 4 college football coaching hire success"
author:
  - name: Michael Schuckers
    affiliation: 
      - name: UNC Charlotte
        department: School of Data Science
        title: Professor of Data Science and Sports Analytics
    email: schuckers@charlotte.edu
  - name: Austin Hayes
    affiliation:
      - name: UNC Charlotte
        title: Data Science Undergraduate
date: "2025-11-20"
abstract: |
  Using data on 103 recent P4 college football hires, we built a statistical model for predicting a coach’s success at their new school[cite: 1]. For each hire, we collected data about their background and experiences, the previous success as a head coach or coordinator and their success since hiring[cite: 1]. Over 50 variables on these factors were recorded though we used 29 of these in building our predictive model[cite: 1]. Our measure of success is based upon Bill Connelly’s SP+ team ratings relative to the performance on the same metric of the school in the 15 year prior to their selection as head coach[cite: 1]. Using a cross-validated regularized linear regression, we obtain a predictive model for coaching success[cite: 1]. Among the important factors for predicting a successful hire are having been a previous college head coach, leaving a job as an Offensive Coordinator, age and quality of the hiring school's team in the previous 15 years[cite: 1]. While we do find these factors are important for the prediction of a successful coaching hire, the trends here are weak[cite: 1]. With 66% accuracy, the model does identify coaching hires that will outperform team performance in the 15 years before the hire[cite: 1]. However, no combination of these factors leads to high predictability of identifying a successful coaching hire[cite: 1]. All of the data and code for this paper are available in a Github repository[cite: 1].
keywords:
  - sports analytics
  - college football
  - regularization
  - predictive modelling
  - machine learning
  - coaching
format:
  html:
    toc: true
    number-sections: true
    code-fold: true
bibliography: sports.bib
---

# Introduction

College football is big business[cite: 1]. The changes due to the use of player's name, image and license (NIL) that allow players to be openly (and appropriately) paid for their services, has brought a new influx of funds[cite: 1]. A December 2024 study by CNBC using fiscal year 2023 numbers estimated that the top university athletic programs are valued at over $1 billion US dollars and have annual revenue of over $200 million [@cnbc_college_worth][cite: 1]. Unsurprisingly the executive that leads the college football operation, the head coach, are well compensated[cite: 1]. In a 2023 study, David Evans found that "80% of the Highest-Earning Public Employees are College Head Coaches" [@highest_state][cite: 1].

Being a successful college football head coach requires multifaceted talent in player development, in gameday decision making, in player evaluation and in recruitment of players[cite: 1]. Consequently, the selection of a new coach for top universities is an important one[cite: 1]. The analysis develops a machine learning predictive model which finds that there are a handful of factors that predict a coaching hire's success[cite: 1]. However, it is clear that our model's predictions are far from certain and that there is a great deal of noise relative to the signals in the data about who would make a successful coach[cite: 1].

In the rest of this paper we look at past hiring of college football head coaches to build a predictive model of the characteristics and factors that influence their success[cite: 1]. The next Section introduces the data that we collected through both automated and manual means[cite: 1]. Summaries of the variables we used as predictors are given in Section 3[cite: 1]. We describe our model building process and results in Section 4[cite: 1]. Finally, we conclude with a discussion of our results[cite: 1].

# Data

The data set used for this analysis is a collection of Power 4 plus Notre Dame (P4) coaching hires[cite: 1]. The P4 conferences are the Atlantic Coast Conference (ACC), Big 12, Big Ten and Southeastern Conference (SEC)[cite: 1]. We began with all coaching hires who started coaching in 2016 or since[cite: 1]. We added at least once coach from each school so that Kirk Ferentz is in our data[cite: 1]. That gave us 120 hires[cite: 1].

Specifically, we include P4 coaching hires who started coaching between 2016 and 2023[cite: 1]. As our observational units, we chose hires rather than coaches since it is the outcome of the hiring process we are trying to maximize[cite: 1]. Consequently, Kirk Ferentz, the current dean of college football, is in the data[cite: 1]. Nick Saban is not among our data, but James Franklin is in the data, as the only Penn State coach over this time period[cite: 1]. Matt Rhule appears two times because of being hired at Baylor (in 2017) and at Nebraska (in 2023)[cite: 1]. We did not include interim coaches and, of course, as the title implies Curt Cignetti is one of our observations and an outlier[cite: 1]. In total there are 103 coaching hires in these data[cite: 1].

For each hire in our data, we recorded over 50 measurements though we used 29 as predictors below[cite: 1]. Generally these fall into three categories: background variables, previous coaching performance, and performance metrics in the job for which they were hired[cite: 1]. Included in the background variables are information about their coaching career, their previous coaching positions, whether or not they have previously coached in the NFL, whether or not they had previously been a HC, whether they had played or coached previously at the hiring institution, etc[cite: 1]. Previous coaching performance variables are recorded and include previous performance as a college head coach[cite: 1]. Variables on a coaches background included features such as whether or not they had a previous connection to the hiring school, whether they played P4 football, and whether they had coached in the NFL[cite: 1]. A full list of all variables is presented in @tbl-var of the Appendix[cite: 1]. One of the metrics we considered was having a connection to the hiring school by being an alumni or having played there or having coached there[cite: 1]. Having an alumni as your head coach was previously analyzed by Josh Planos of FiveThirtyEight and found to not impact coaching success either way [@alumni_538][cite: 1]. To account for regression to the mean, a team that was underperforming relative to historical averages before a coach arrived, we also included the team's average performance for the five years before they arrived[cite: 1].

::: {#fig-cat1 layout-ncol=2}
![Regional Ties](img/cfb_sp_vs_RegionalTies.png){#fig-reg-ties}

![Prior Connection to School](img/cfb_sp_vs_Connection.png){#fig-connection}

![Previously won Conf. Champ. as HC](img/cfb_sp_vs_PrevCCasHC.png){#fig-prev-cc}

![Previously won Natl. Champ. as Coordinator](img/cfb_sp_vs_PrevNCasCoord.png.png){#fig-prev-nc}

![Previous Job with P4 School](img/cfb_sp_vs_LeavingP4.png){#fig-p4}

![Previous Job with NFL](img/cfb_sp_vs_LeavingNFL.png){#fig-nfl}

![Previous Job as OC](img/cfb_sp_vs_LastOC.png){#fig-oc}

![Previously been HC any level college](img/cfb_sp_vs_PrevHCAny.png){#fig-hc-any}

Plots for selected categorical predictors vs SP Rel[cite: 1]
:::

To determine how successful each coach was as a hire we calculated their performance as a HC relative to the historical performance of the university where they were hired[cite: 1]. For performance we considered three measurements[cite: 1]. First we considered a hire's win percentage since winning is arguably the most important outcome for a head coach[cite: 1]. Second, we considered a coach's average Simple Rating System (SRS)[cite: 1]. This metric, SRS, is a "weighted scoring margin" where the weighted part is based upon the quality of the teams that have been played [@intro_srs][cite: 1]. Third, we considered a coaches average relative SP+ which is a predictive metric developed by Bill Connelly [@intro_sp] that is a "tempo- and opponent-adjusted measure" of team efficiency [@sp_2025][cite: 1]. The historical values for win percentage, SRS and SP+ were obtained using the `cfbfastr` package [@cfbfastr][cite: 1]. We augmented those data with wins and losses from the 2025 season[cite: 1].

| Name | Hiring School | *SP+ Rel* |
|:---|:---|---:|
| Josh Heupel | UCF | 22.6 |
| Rhett Lashlee | SMU | 17.5 |
| Sonny Dykes | SMU | 16.1 |
| Mike Gundy | Oklahoma State | 13.8 |
| Mike Elko | Duke | 10.7 |
| Matt Campbell | Iowa State | 9.9 |
| Dabo Swinney | Clemson | 9.7 |
| Bret Bielema | Illinois | 9.5 |
| Lane Kiffin | Ole Miss | 8.8 |
| Josh Heupel | Tennessee | 8.7 |

: Best Performing Coaching Hires by *SP+ Rel*. {#tbl-topcoaches}[cite: 1]

One of the significant challenges for this type of analysis is to define success[cite: 1]. This is certainly the case here, and in no small part is due to the expectations for each school[cite: 1]. What might be a successful record at Arizona, might not be a success at Alabama[cite: 1]. What gets a coach an extension in College Park, might be a reason to be fired in College Station or State College[cite: 1]. Consequently, our definition of coaching performance is relative (to team history) rather than absolute[cite: 1].

For relative performance, we took the trimmed mean (`mean(x, trim=0.1)` in R) of the middle 80% of years of the performance metric for their time as a head coach and subtracted the team average of that same metric for the fifteen years before the coach was hired[cite: 1]. We chose to use SP+ as our measure and we refer to *SP+ Rel* as our outcome variable below[cite: 1]. Note that the correlation between SRS relative to past team performance and SP+ relative to past team performance was 0.95, results using the former as a response are likely to be very similar to those for the latter and so we did not consider models with different responses[cite: 1]. The correlation between *SP+ Rel* and a winning percentage relative to past teams was 0.82[cite: 1].

Using the R package `cfbfastr`, we obtained historical team data including wins and losses through the 2024 season as well as SRS and SP+ ratings into the 2025 season[cite: 1]. We manually added wins and losses for the 2025 season[cite: 1]. All of these data can be found at: [https://github.com/schuckers/cfb_coach_analytics](https://github.com/schuckers/cfb_coach_analytics)[cite: 1]. Data on coaching performance is complete through midday on November 13, 2025[cite: 1].

# Data Summaries

Among these 103 college football head coaching hires in our data, there were 50 who left a job as head coach (HC) at another school, and 63 of our hires had been a HC at some level previous to their being hired[cite: 1]. Twenty-eight of the hired coaches had been Defensive Coordinators (DCs) and 15 were Offensive Coordinators (OCs)[cite: 1]. The other previous positions included four position coaches, an offensive analyst, four Associate or Assistant HCs and one hire who was not employed in football[cite: 1]. While more coaches were hired directly from DC positions, 40 of the 103 hires (39%) had defense as their background[cite: 1]. Sixty-three of the hires had been a head coach at the college or NFL previously and in total only 24 had previously coached in the NFL[cite: 1]. Thirty-one percent of the coaches were hired to jobs within the same conference[cite: 1]. Only 16 of our hires were promoted from the coaching staff of the same institution including Dabo Swinney[cite: 1]. The average age of our coaches when they were hired is approximately 46 years[cite: 1].

The majority of hired coaches, 65, had not worked for, played at or attended the school that hired them[cite: 1]. While the vast majority of the hired coaches played college football, just under 50% of them, 49, played for a P4 school and only ten played in the NFL[cite: 1].

Below when we refer to recent metrics we will be talking about a five year period and when we refer to previous metrics we will be talking about a 15 year period[cite: 1].

::: {#fig-numeric1 layout-ncol=2}
![Numb. years as Coord or HC](img/cfb_sp_vs_numb_yrs_Coord_HC_quad.png){#fig-yrs-coord}

![Age - 50](img/cfb_sp_vs_Age.png){#fig-age}

![Teams SP+ in last 5 years](img/cfb_sp_vs_TeamSPRecent.png){#fig-sp-recent}

![Teams SP+ in last 15 years](img/cfb_sp_vs_TeamSP_Prev15.png){#fig-sp-prev15}

Plots for selected numeric predictors vs SP Rel[cite: 1]
:::

Next we will look at some of relationship between our variables and our measure of coaching performance, *SP+ Rel*[cite: 1]. @tbl-var contains the list of predictor variables as well as their definitions[cite: 1]. Turning to the graphics, we see a sampling of these relationship in @fig-cat1[cite: 1]. In that figure, we can see that medians (red lines) seem to indicate there are some differences between the categories for winning a national championship as a coordinator, @fig-prev-nc, and for previous job was as an OC, @fig-oc[cite: 1]. Otherwise the other variables do not seem to suggest there is much difference in the typical coaching performance for the different characteristics recorded in those subfigures[cite: 1]. More graphics like these can be found for the other categorical predictors in @fig-append-cat and @fig-append-cat2[cite: 1]. In those graphics, @fig-append-cat (d) suggests that having played in the NFL yields a higher median *SP+ Rel* as does having previously won a national championship as a coordinator, @fig-append-cat2 (b)[cite: 1]. However, quite a bit fewer coaches have the value that is associated with the higher median[cite: 1].

Turning to numeric predictors, we can see the relationship between *SP+ Rel* and four of these predictors in @fig-numeric1[cite: 1]. In @fig-yrs-coord, we can see that there is not a relationship between number of years as a HC or Coordinator before being hired[cite: 1]. The three remaining subfigures, @fig-age, @fig-sp-recent, @fig-sp-prev15, also show negative linear relationships between *SP+ Rel* and coach's age, teams performance in the recent past, 5 years, and team performance in the previous 15 years[cite: 1]. More figures like this can be found in @fig-append-numeric of the Appendix at the end of this report[cite: 1]. Noteworthy among those figures in the Appendix is that there seems to be a positive linear relationship between a hired coach's *SP+ Rel* and their previous *SP+ Rel* had they been a college HC previously[cite: 1].

# Model and Results

The goal of this analysis is to build a predictive model for the success of a coaching hire and to identify the factors that impact that success[cite: 1]. We selected 29 predictors found in @tbl-var that could be useful for predicting coaching success[cite: 1]. For our model formulation, we chose to build a regularized linear regression model for SP+ relative to past team performance, hereafter *SP+ Rel*[cite: 1]. We used `cv.glmnet` as part of the `glmnet` library in R [@glmnet][cite: 1]. For our regularization, we considered both ridge and lasso regression, and we used 8-fold cross validated performance to determine the regularization/shrinkage parameter, $\lambda$, for our model[cite: 1]. We ran 20 different randomizations of 8-fold cross validation[cite: 1]. Consistently, across all these different randomizations of the folds, we found that the best lasso model had a smaller out of sample prediction error (via MSE) than the best ridge model[cite: 1]. To get the parameter coefficient estimates discussed below we fit the full model via a lasso with $\lambda$ set to the average of the shrinkage parameters that minimized the cross-validation error[cite: 1]. In fact there was no variability in these shrinkage parameters across the 20 randomizations[cite: 1]. Because some of our predictors, e.g. `sp_prev_rel`, were available only if a coach had previously been a college HC, we included whether or not a coach was previously an HC in all of the models we considered[cite: 1].

Our final prediction equation is:

$$\begin{aligned} \widehat{\text{SP+ Rel}} = -0.054 &+ 0.493 \, (\text{Was College HC before at any level}) \\ &+ 3.750 \, (\text{Previous job was as an OC}) \\ &- 0.002 \, (\text{Age} - 50) \\ &- 0.307 \, (\text{Hiring School's Team SP+ in 15 years before hiring}) \end{aligned}$$
{#eq-predictioneq}

The largest factors in our model were whether or not the coach's previous job was as an OC (`PreviousOC`), whether they had ever been a college coach previously (`prevCollegeHC`) and the team's performance over the last 15 years (`team_sp_prev`)[cite: 1]. Having been hired from an OC position is associated with about a $3.75$ point increase in a coach's *SP+ Rel*[cite: 1]. While if a hire was previously a college football HC, the predicted increase in their team's performance increases by nearly half a point[cite: 1]. Age is a factor in our model that is negatively correlated with coaching success though the effect is very small, about two-hundredth of a point per game for an additional 10 years of age[cite: 1]. This seems quite negligible[cite: 1]. For a newly hired 50 year old coach on a team whose SP+ averaged 0 in the last 15 years with none of the other characteristics in the model, we expect their *SP+ Rel* to be -0.054 or just slightly below what the team had done in the previous 15 years[cite: 1].

The last term in our prediction equation involves a team's average *SP+ Rel* over the previous 15 years and that is negatively associated with how a coaching hire's teams will perform[cite: 1]. We can interpret the value here of $-0.307$ as the expected decrease in a team's average performance for every additional point of *SP+ Rel* that the hiring team had over the previous 15 years[cite: 1]. This is a regressive effect, since SP+ for an average team should be around zero, then for a new coaching hire, good teams we would expect them to get worse and very good teams we would expect them to get much worse, while for below average teams we would expect them to get better just by making the hire[cite: 1]. One possible explanation for this is a regression to the mean as team performance tends to move to the average of all teams, which for SP+ is around zero[cite: 1]. Another explanation is college football has moved to an era of more parity and relative coaching performance is not immune to moving with that trend[cite: 1].

The hires with the ten highest and ten lowest predicted *SP+ Rel* are listed in @tbl-topcoaches-pred[cite: 1]. We can see that the model suggests that the hire with the highest predicted success was Rhett Lashlee at SMU[cite: 1]. Eight of the ten top predicted *SP+ Rel* values had positive observed values[cite: 1]. Tony Elliott at Virginia and Jedd Fisch at Arizona were the exceptions[cite: 1]. Elliott has improved his Virginia teams and, if he stays, may yet get to positive territory[cite: 1]. Fisch left Arizona after a successful second season[cite: 1]. Among the coaches predicted to have negative *SP+ Rel*, eight of the ten are in or have finished in negative territory[cite: 1]. The exceptions are Dan Mullen whose team performance dropped significantly over his four years at Florida and Kirby Smart who has impressively elevated Georgia to new heights[cite: 1]. Six of these ten coaches have already been fired with only Mullen, Smart, Fickell and Venables still employed at these schools[cite: 1]. Venables' Oklahoma 2025 team is having their best season with him as a HC and so there is hope for a turnaround there[cite: 1]. Looking at the bottom ten schools, most of the schools there are considered to be 'elite' or nearly so[cite: 1]. The model predicts that succeeding at these schools is hard partly because they are already at the upper tail of the distribution[cite: 1]. Overall the model accurately predicts whether a coach will have a positive or negative *SP+ Rel* 67% of the time[cite: 1].

| Name | Hiring School | *SP+ Rel* | $\widehat{\text{SP+ Rel}}$ |
|:---|:---|---:|---:|
| Rhett Lashlee | SMU | 17.8 | 6.3 |
| Josh Heupel | UCF | 22.6 | 5.7 |
| Major Applewhite | Houston | 6.7 | 4.2 |
| Mike Gundy | Oklahoma State | 13.8 | 4.2 |
| Sonny Dykes | SMU | 16.1 | 4.1 |
| Tony Elliott | Virginia | -5.6 | 3.5 |
| Mike Locksley | Maryland | 0.4 | 2.9 |
| Jedd Fisch | Arizona | -5.3 | 2.7 |
| Joe Moorhead | Mississippi State | 8.5 | 2.7 |
| Lance Leipold | Kansas | 8.5 | 2.3 |
| **---** | **---** | **---** | **---** |
| Dan Mullen | Florida | 1.1 | -5.7 |
| Mark Richt | Miami | -2.4 | -5.8 |
| Luke Fickell | Wisconsin | -15.0 | -5.9 |
| Tom Herman | Texas | -9.3 | -6.2 |
| Kirby Smart | Georgia | 4.5 | -6.4 |
| Willie Taggart | Florida State | -19.1 | -6.4 |
| Jim McElwain | Florida | -8.3 | -6.6 |
| Brian Kelly | LSU | -6.0 | -6.7 |
| Brent Venables | Oklahoma | -8.9 | -6.8 |
| Ed Orgeron | LSU | -4.8 | -7.4 |

: Top 10 & Bottom 10 Coaching Hires by Predicted *SP+ Rel*. {#tbl-topcoaches-pred}[cite: 1]

The model we have explains about $24\%$ of the variability in the team's performance with a newly hired coach and the average difference between our model's prediction and the actual *SP+ Rel* of the coaches in these data is just over $7$ points[cite: 1]. In 2025 terms, that difference is akin to the average difference in weekly performance between Texas A&M and Missouri, or between USC and Nebraska or between Georgia Tech and NC State[cite: 1]. That is a large amount of variability and it speaks to the noisiness of these data[cite: 1]. However, the model does well given that noisiness at predicting who will have a *SP+ Rel* above or below zero achieving 66% accuracy on this metric[cite: 1].

While it is important to focus on the factors that do influence hiring success, it is worth mentioning those that do not given the factors that **are** in the model[cite: 1]. Having a previous connection to the school, being a previous head coach at any level, having NFL coaching experience, having experience as a recruiting coordinator, having played football in the NFL or at a P4 school seem to have no impact on our ability to predict how well a coach will do[cite: 1]. Winning a national championship as a HC at any level as well as being a Coordinator for a team that won a national championship or a conference championship does not predict coaching success[cite: 1]. Similarly having regional ties to the hiring school does not seem to be impactful and the same is true for moving within the same conference or leaving a P4 school[cite: 1]. A coach’s relative SP+ as a previous HC is also not predictive, as is their previous win percentage[cite: 1]. One possible explanation is that we are taking averages of previous coaching performance that misses changes in SP+ over a coach's time with a previous school, e.g. Matt Rhule at Temple and Lance Leipold at Buffalo[cite: 1]. Finally, the number of games coached as an HC before being hired is unimportant as a predictor as is the recent winning percentage of the team[cite: 1].

# Discussion

Choosing a successful P4 college football coach is a difficult task[cite: 1]. In this paper we have collected data on recent P4 coaching hires and built a predictive model for their performance[cite: 1]. The data we used was on recent P4 hires and we measured their performance using average SP+ relative to previous performance of the school’s teams, *SP+ Rel*[cite: 1]. As mentioned above *SP+ Rel* is correlated with a coach's winning percentage and highly correlated with their SRS, the other statistical measure of team performance that we considered[cite: 1].

Ultimately, we found that there are several factors that are predictive of the performance of a college football head coaching hire[cite: 1]. These factors include whether a hire had been a previous college coach, whether the hire had previously won a conference championship as a coach, whether their prior job was as an offensive coordinator, their age and the quality of the team they were hired to lead over the previous 15 years[cite: 1]. Our prediction model does not perform particularly well, only explaining just under a quarter of the variability in coaching performance[cite: 1].

The 2020 season is among the years in these data[cite: 1]. The effects of COVID were felt across college football in the form of shortened seasons and empty stadiums[cite: 1]. Every school and every conference handled the situation differently[cite: 1]. One of the choices we made as part of this analysis was that we would keep the 2020 season in our data[cite: 1]. Given the short tenure of some college football head coaching hires, we felt having as much data as possible was important[cite: 1].

This study measured the success of a coaching hire by how they did on average relative to their team's previous 15 years[cite: 1]. That is not a perfect measure[cite: 1]. Team performance fluctuates and trends, sometimes upward and sometimes downward[cite: 1]. Ideally when a new coach is hired their teams trend upward as they recruit players that fit their system and they become comfortable in their new job[cite: 1]. An average over a short period may miss this[cite: 1]. That said, it is very difficult to generate a measure of coaching success that captures all of the possible nuances[cite: 1]. Therefore, we chose to stick with an average relative to past team performance[cite: 1].

Much has been made about the recent success of Curt Cignetti and he has done an amazing job as the HC at Indiana[cite: 1]. So it is worth mentioning that Cignetti is not in the data we used for modelling because we have less than two full years of him coaching at Indiana[cite: 1]. Regardless, Cignetti is an outlier[cite: 1]. It is possible to regenerate all of the figures in this analysis with Cignetti and other coaches hired after the 2023 season and it is very clear which observation corresponds to the exceptional job Cignetti has done at Indiana[cite: 1]. That said, an outlier is not a trend and the signal from these data is weak[cite: 1]. Hiring a coach is hard[cite: 1]. There are no magic attributes, at least *not* among the factors we investigated, that guarantee future success[cite: 1]. The data as can be seen in @fig-cat1 through @fig-append-numeric is very noisy and the trends, while tempting, are mild[cite: 1]. Great coaches are hard to find and many schools find themselves waiting for their next great coach[cite: 1].

The code and the data for the analyses presented here can be found at [https://github.com/schuckers/cfb_coach_analytics](https://github.com/schuckers/cfb_coach_analytics)[cite: 1].

Thanks to the team of folks who put together `cfbfastr` for making their data available in a usable and regularly updated manner and to Brian Macdonald (Yale) for thoughtful discussions about this analysis[cite: 1].

# Appendix

| Name | Hiring School | *SP+ Rel* |
|:---|:---|---:|
| Troy Taylor | Stanford | -22.3 |
| Willie Taggart | Florida State | -19.1 |
| Geoff Collins | Georgia Tech | -18.5 |
| Chad Morris | Arkansas | -18.0 |
| Luke Fickell | Wisconsin | -15.0 |
| Karl Dorrell | Colorado | -14.2 |
| Jeff Hafley | Boston College | -12.6 |
| Brent Pry | Virginia Tech | -11.8 |
| Ryan Walters | Purdue | -11.7 |
| Mel Tucker | Michigan State | -11.5 |

: Worst Performing Coaching Hires by *SP+ Rel*. {#tbl-bottomcoaches}[cite: 1]

| Variable Name | Variable Meaning |
|:---|:---|
| `prevCollegeHC` | Were they previously a college head coach (HC) |
| `PreviousHC` | Was job when hired as a HC |
| `PreviousOC` | Was job when hired as an Offensive Coordinator (OC) |
| `PreviousDC` | Was job when hired as a Defensive Coordinator (DC) |
| `Conf.LeavingP4` | Was job when hired at P4 school |
| `SameConf` | Was job when hired in same conference |
| `AgeAtHiringMinus50` | Age on Jan 1 of first season as HC |
| `ConnectionToHiring.School` | Had coach previously played or been employed at hiring school |
| `NFLCoachingExperience` | Had the coach ever coached in the NFL |
| `RecruitingCoord` | Previously been a recruiting coordinator or not |
| `PlayedNFL` | Had coach played in the NFL |
| `Played.Power.4` | Had coach played at a Power 4 school |
| `WonNCasHCany` | Had coach won a National Championship (NC) as HC at any level (college) |
| `WonCCasHCany` | Had coach won a Conference Championship (CC) as HC at any level (college) |
| `WonNCasCoordAny` | Had coach won a NC as an OC, DC, or AHC at any level (college) |
| `WonCCasCoordAny` | Had coach won a CC as an OC, DC, or AHC at any level (college) |
| `SideO` | Is coach's primary background on Offense |
| `LeavingNFL` | Was previous job in the NFL |
| `Regional.Ties` | Had coach previously worked within 500 miles of hiring school |
| `numb_years_Coord_adj` | Number of years as OC/DC/AHC/HC above average of 11.8 years |
| `numb_years_Coord_HCadj2` | Numbers of years as OC/DC/AHC/HC different from 11.8 squared |
| `PromoteWithin` | Coach was promoted to HC from other position at same school |
| `AllPriorWinPct` | Win Percent as Coach at all levels before being hired |
| `winpct_prev_Rel` | Win percent as HC at previous job |
| `team_sp_prev` | Hiring school's SP+ in 15 years before hire |
| `team_sp_recent` | Hiring school's SP+ in 5 years before hire |
| `GamesAsHCPrior` | Number of games as a HC before being hired |
| `team_winpct_recent` | Hiring School's win percentage in the last 5 years |
| `sp_prev_rel` | Previous *SP+ Rel* in HC role |

: List of variables used as predictors in this paper. All values are relative to the coach at the time they were hired. {#tbl-var}[cite: 1]

::: {#fig-append-cat layout-ncol=2}
![Previous job as DC](img/cfb_sp_vs_LastDC.png){#fig-last-dc}

![Previous job as HC](img/cfb_sp_vs_PreviousHC.png){#fig-prev-hc}

![Played P4 football](img/cfb_sp_vs_PlayedP4.png){#fig-played-p4}

![Played in the NFL](img/cfb_sp_vs_PlayedNFL.png){#fig-played-nfl}

![Side of the Ball](img/cfb_sp_vs_SideOfBall.png){#fig-side-ball}

![Promoted from Within](img/cfb_sp_vs_PromoteWithin.png){#fig-promote-within}

Additional plots of categorical predictors vs *SP+ Rel*[cite: 1]
:::

::: {#fig-append-cat2 layout-ncol=2}
![Has NFL coaching experience](img/cfb_sp_vs_PrevNFLCoaching.png){#fig-nfl-exp}

![Won Natl. Champ. as HC any level](img/cfb_sp_vs_WonNCasHCAny.png){#fig-nc-hc}

![Won Conf. Champ. as Coord any level](img/cfb_sp_vs_WonCCasCoordAny.png){#fig-cc-coord}

![Hired within same conference](img/cfb_sp_vs_SameConf.png){#fig-same-conf}

![Been Recruiting Coordinator previously](img/cfb_sp_vs_PrevRecruitCoord.png){#fig-recruit-coord}

Additional plots of categorical predictors vs *SP+ Rel*[cite: 1]
:::

::: {#fig-append-numeric layout-ncol=2}
![All Past Win Pct as HC](img/cfb_sp_vs_AllPastWinPctAsHC.png){#fig-all-winpct}

![Win Pct at Previous job if HC](img/cfb_sp_vs_Coach_PrevWinPct.png){#fig-prev-winpct}

![Coach *SP+ Rel* before hire](img/cfb_sp_vs_Coach_sp_rel_before.png){#fig-coach-sp-before}

![Numb. of games as HC](img/cfb_sp_vs_numb_gamesAsHC.png){#fig-games-hc}

![Team Win Pct in last 5 years](img/cfb_sp_vs_TeamWinPctRecent.png){#fig-team-winpct-recent}

Additional plots comparing numeric predictors and *SP+ Rel*[cite: 1]
:::

| | Actual: *SP+ Rel* < 0 | Actual: *SP+ Rel* > 0 |
|:---|---:|---:|
| **Predicted: $\widehat{\text{SP+ Rel}} < 0$** | 51 | 23 |
| **Predicted: $\widehat{\text{SP+ Rel}} > 0$** | 11 | 18 |

: Truth Table for *SP+ Rel*. {#tbl-truth}[cite: 1]
