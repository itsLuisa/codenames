setwd("C:/Users/Luisa/Documents/Uni/SS 2021/Bachelorarbeit/codenames")

#baseline
baselinezis <- read.csv("results/baseline_ZiS.csv", sep = ";")
mean(baselinezis$rank)
sum(baselinezis$score)
baselineeval<- read.csv("results/baseline_eval.csv", sep = ";")
mean(baselineeval$rank)
sum(baselineeval$score)
baselineall <- rbind(baselinezis, baselineeval)
mean(baselineall$rank)
sum(baselineall$score)

# bert
glosszis <- read.csv("results/gloss_ZiS.csv", sep=";", header=TRUE)
glosseval <- read.csv("results/gloss_eval.csv", sep=";", header=TRUE)

percent1 <- df_all$score_bert / df_all$clue_no
percent2 <- df_all$score_gloss / df_all$clue_no
t.test(df_all$score_bert, df_all$score_gloss, paired=TRUE)
t.test(percent1, percent2, paired=TRUE)
t.test(df_all$rank_bert, df_all$rank_gloss, paired = TRUE)
better <- df_all$rank_gloss < df_all$rank_bert
better
lmer(better ~ df_all$rank_gloss + (1|item), family(binomial))

# rsa
cluefinderzis <- read.csv("results/cluefinder_ZiS_rsa.csv", sep = ";")
mean(cluefinderzis$rank)
sum(cluefinderzis$score)
cluefindereval <- read.csv("results/cluefinder_eval_rsa.csv", sep = ";")
mean(cluefindereval$rank)
sum(cluefindereval$score)
cluefinderall <- rbind(cluefinderzis, cluefindereval)
mean(cluefinderall$rank)
sum(cluefinderall$score)
googlezis <- read.csv("results/google_ZiS_rsa.csv", sep = ";")
mean(googlezis$rank)
sum(googlezis$score)
googleeval <- read.csv("results/google_eval_rsa.csv", sep = ";")
mean(googleeval$rank)
sum(googleeval$score)
googleall <- rbind(googlezis, googleeval)
mean(googleall$rank)
sum(googleall$score)
embeddingszis <- read.csv("results/embeddings_ZiS_rsa.csv", sep = ";")
mean(embeddingszis$rank)
sum(embeddingszis$score)
embeddingseval <- read.csv("results/embeddings_eval_rsa.csv", sep = ";")
mean(embeddingseval$rank)
sum(embeddingseval$score)
embeddingsall <- rbind(embeddingszis, embeddingseval)
mean(embeddingsall$rank)
sum(embeddingsall$score)
filteredzis <- read.csv("results/filtered_ZiS_rsa.csv", sep = ";")
mean(filteredzis$rank)
sum(filteredzis$score)
filteredeval <- read.csv("results/filtered_eval_rsa.csv", sep = ";")
mean(filteredeval$rank)
sum(filteredeval$score)
filteredall <- rbind(filteredzis, filteredeval)
mean(filteredall$rank)
sum(filteredall$score)
singlezis <- read.csv("results/single_ZiS_rsa.csv", sep = ";")
mean(singlezis$rank)
sum(singlezis$score)
singleeval <- read.csv("results/single_eval_rsa.csv", sep = ";")
mean(singleeval$rank)
sum(singleeval$score)
singleall <- rbind(singlezis, singleeval)
mean(singleall$rank)
sum(singleall$score)

filteredzisshort <- read.csv("results/filtered_ZiS_rsa_short.csv", sep = ";")
filteredevalshort <- read.csv("results/filtered_eval_rsa_short.csv", sep = ";")
mean(filteredevalshort$rank)
sum(filteredevalshort$score)
sum(filteredevalshort$clue_no)
filteredallshort <- rbind(filteredzisshort, filteredevalshort)
mean(filteredallshort$rank)
sum(filteredallshort$score)
glossrsazis <- read.csv("results/gloss_ZiS_rsa.csv", sep=";")
glossrsaeval <- read.csv("results/gloss_eval_rsa.csv", sep=";")
mean(glossrsaeval$rank)
sum(glossrsaeval$clue_no)
sum(glossrsaeval$score)
glossrsaall <- rbind(glossrsazis, glossrsaeval)
mean(glossrsaall$rank)

hist(embeddingsall$score, breaks = 50)
shapiro.test(embeddingsall$score)
shapiro.test(embeddingsall$rank)

baselineall$score
cluefinderall$score
data <- data.frame(baselinezis$score, cluefinderzis$score)
library(tidyr)
new_data <- gather(data, model, score, baselinezis.score:cluefinderzis.score)
new_data$model <- as.factor(new_data$model)
t.test(score~model, data = new_data)

library(ggplot2)
ggplot(study, aes(x=board_size, y=baseline_rank)) +
  geom_point() +
  geom_smooth(formula = y ~ x)

# kunar eval
kunar_glove <- read.csv("results/kunar_baseline.csv", sep = ";")
kunar_glove_rsa <- read.csv("results/kunar_rsa_glove.csv", sep = ";")
summary(kunar_glove)
m = mean(kunar_glove$top.5.acc)
m = mean(kunar_glove$rank)
m = mean(kunar_glove_rsa$top.5.acc)
m = mean(kunar_glove_rsa$rank)
s = sd(kunar_glove_rsa$rank)
n = 1025
margin <- qt(0.975,df=n-1)*s/sqrt(n)
m
m - margin
m + margin

# study eval
raw_study <- read.csv("eval_study/experiment_raw_data.csv", sep = ",")
operativesbaseline <- read.csv("eval_study/study_operatives_baseline.csv", sep=";")
sum(operativesbaseline$clue_no)
sum(operativesbaseline$score)
mean(operativesbaseline$rank)
spymastersbaseline <- read.csv("eval_study/study_spymasters_baseline.csv", sep=";")
sum(spymastersbaseline$clue_no)
sum(spymastersbaseline$score)
mean(spymastersbaseline$rank)
sum(operativesbaseline$score) + sum(spymastersbaseline$score)
mean(rbind(operativesbaseline$rank, spymastersbaseline$rank))

operativesgloss <- read.csv("eval_study/study_operatives_gloss.csv", sep = ";")
sum(operativesgloss$score)
spymastersgloss <- read.csv("eval_study/study_spymasters_gloss.csv", sep = ";")

operativesglossRSA <- read.csv("eval_study/study_operatives_gloss_rsa.csv", sep = ";")
spymasterglossRSA <- read.csv("eval_study/study_spymasters_gloss_rsa.csv", sep = ";")

operativesbaselineRSA <- read.csv("eval_study/study_operatives_baseline_rsa.csv", sep = ";")

# fleiss kappa
library(irr)
data("diagnoses", package = "irr")
head(diagnoses[, 1:3])
kappam.fleiss(diagnoses[,1:3])
mydata
example <- c("house", "trash", "bit")
ex <- c("house", "trash", "else")
ex2 <- c("house", "bla", "else")
hello <- data.frame(example, ex, ex2)
kappam.fleiss(hello)
annotators <- read.csv("eval_study/interannotatoragreement.csv", sep = ";")
annotators$participant <- as.factor(annotators$participant)
annotators$guess <- as.factor(annotators$guess)
annotators_wide <- spread(annotators, participant, guess)
aa <- annotators_wide[4:29]
kappam.fleiss(annotators_wide)



#Montag 13.12.
operativesbaseline$score_normalized <- operativesbaseline$score / operativesbaseline$clue_no
operativesbaseline$participants_no <- c(4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,2,2,2,2,2,5,5,5,5,5,3,3,3,3,3,1,1,1,1,1,3,3,3,3,3)
spymastersbaseline$score_normalized <- spymastersbaseline$score / spymastersbaseline$clue_no

n_guesses_given <- c(2,3,3,3,1,3,2,3,6,3,9,3,2,4,4,1,3,2,2,1,6,3,1,3,4,2,3,3,3,3,3,2,2,2,2,2,5,3,3,2)
operativesbaseline$n_guesses_given <- n_guesses_given
operativesbaseline$n_guesses_given_normalized <- operativesbaseline$n_guesses_given / operativesbaseline$clue_no

operativesbaselineadjusted <- subset(operativesbaseline, clue != "chopping" & clue != "einstein")
#baseline_together <- data.frame(operativesbaseline$score, spymastersbaseline$score, operativesbaseline$rank, spymastersbaseline$rank)

operativesbaselineRSA$score_normalized <- operativesbaselineRSA$score / operativesbaselineRSA$clue_no
operativesbaselineRSA$n_guesses_given <- n_guesses_given
operativesbaselineRSA$n_guesses_given_normalized <- operativesbaselineRSA$n_guesses_given / operativesbaselineRSA$clue_no
operativesbaselineRSA$additional_given_guesses <- operativesbaselineRSA$n_guesses_given - operativesbaselineRSA$clue_no

operativesbaselineRSAadjusted <- subset(operativesbaselineRSA, clue != "chopping" & clue != "einstein")

operativesgloss$score_normalized <- operativesgloss$score / operativesgloss$clue_no
operativesgloss$participants_no <- c(4,4,4,4,4,4,4,4,4,4,4,4,4,4,2,2,2,2,2,5,5,5,5,5,3,3,3,3,3,1,1,1,1,3,3,3,3,3)
spymastersgloss$score_normalized <- spymastersgloss$score / spymastersgloss$clue_no

operativesglossRSA$score_normalized <- operativesglossRSA$score / operativesglossRSA$clue_no
operativesglossRSA$participants_no <- c(4,4,4,4,4,4,4,4,4,4,4,4,4,4,2,2,2,2,2,5,5,5,5,5,3,3,3,3,3,1,1,1,1,3,3,3,3,3)
spymasterglossRSA$score_normalized <- spymasterglossRSA$score / spymasterglossRSA$clue_no

# boxplots to keep
operativesranks <- data.frame(operativesbaselineadjusted$rank, operativesbaselineRSAadjusted$rank, operativesgloss$rank, operativesglossRSA$rank)
names(operativesranks) <- c("baseline", "baseline + RSA", "GlossBERT", "GlossBERT + RSA")
boxplot(operativesranks)
summary(operativesranks)

operativesscores <- data.frame(operativesbaselineadjusted$score_normalized, operativesbaselineRSAadjusted$score_normalized, operativesgloss$score_normalized, operativesglossRSA$score_normalized)
names(operativesscores) <- c("baseline", "baseline + RSA", "GlossBERT", "GlossBERT + RSA")
boxplot(operativesscores)
summary(operativesscores)

alloperativesrankandaddguesses <- data.frame(operativesbaselineadjusted$rank, operativesbaselineRSAadjusted$rank, operativesgloss$rank, operativesglossRSA$rank, operativesbaselineRSAadjusted$additional_given_guesses)
names(alloperativesrankandaddguesses) <- c("baseline", "baselineRSA", "GlossBERT", "GlossBERTRSA", "additional_guesses")
longaddguess <- gather(alloperativesrankandaddguesses, model, rank, baseline:baselineRSA:GlossBERT:GlossBERTRSA)

m = mean(operativesranks$`GlossBERT + RSA`)
s = sd(operativesranks$`GlossBERT + RSA`)
n = 1025
margin <- qt(0.975,df=n-1)*s/sqrt(n)
m
m - margin
m + margin

# not much information here
spymastersglossscores <- data.frame(spymastersgloss$score_normalized, spymasterglossRSA$score_normalized)
names(spymastersglossscores) <- c("GlossBERT", "GlossBERT + RSA")
boxplot(spymastersglossscores)
spymasterglossranks <- data.frame(spymastersgloss$rank, spymasterglossRSA$rank)
names(spymasterglossranks) <- c("GlossBERT", "GlossBERT + RSA")
boxplot(spymasterglossranks)


#analyze with normalized score values:

score_data <- data.frame(operativesbaselineadjusted$score_normalized, operativesgloss$score_normalized)

analysis_data <- gather(score_data, model, score, operativesbaselineadjusted.score_normalized:operativesgloss.score_normalized)
analysis_data$model <- as.factor(analysis_data$model)

t.test(score~model, data = analysis_data)
sd(operativesgloss$score_normalized)
sd(operativesbaselineadjusted$score_normalized)


#analyze without normalize score values:

score_data_unnormalized <- data.frame(studyoperativesbaselineadjusted$score, studyoperativesgloss$score)

analysis_data_unnormalized <- gather(score_data_unnormalized, model, score, studyoperativesbaselineadjusted.score:studyoperativesgloss.score)
analysis_data_unnormalized$model <- as.factor(analysis_data_unnormalized$model)

t.test(score~model, data = analysis_data_unnormalized)


#Wilcoxon-Test for ranking:

ranking_data <- data.frame(studyoperativesbaselineadjusted$rank, studyoperativesgloss$rank)

analysis_data_ranking <- gather(ranking_data, model, rank, studyoperativesbaselineadjusted.rank:studyoperativesgloss.rank)
analysis_data_ranking$model <- as.factor(analysis_data_ranking$model)

mean(studyoperativesbaselineadjusted$rank)
mean(studyoperativesgloss$rank)

wilcox.test(rank~model, data = analysis_data_ranking, exact = FALSE, correct = FALSE, conf.int = FALSE)


#Jetzt kommt noch RSA dazu:
#studyoperativesglossRSA <- read.csv("results/study_operatives_gloss_rsa.csv", sep = ";")

#studyoperativesglossRSA$score_normalized <- studyoperativesglossRSA$score / studyoperativesglossRSA$clue_no
operativesbaselineRSAadjusted$participants_no <- c(4,4,4,4,4,4,4,4,4,4,4,4,4,4,2,2,2,2,2,5,5,5,5,5,3,3,3,3,3,1,1,1,1,3,3,3,3,3)


score_data_all <- data.frame(operativesbaselineadjusted$score_normalized, operativesbaselineRSAadjusted$score_normalized, operativesgloss$score_normalized, operativesglossRSA$score_normalized)

analysis_data_all <- gather(score_data_all, model, score, operativesbaselineadjusted.score_normalized:operativesbaselineRSAadjusted.score_normalized:operativesgloss.score_normalized:operativesglossRSA.score_normalized)
analysis_data_all$model <- as.factor(analysis_data_all$model)

anova <- aov(score ~ model, data = analysis_data_all)
summary(anova)

#Unterschied zwischen Gloss und GlossRSA:

score_data_GlossVSRSA <- data.frame(operativesgloss$score_normalized, operativesglossRSA$score_normalized)

analysis_data_GlossVSRSA <- gather(score_data_GlossVSRSA, model, score, operativesgloss.score_normalized:operativesglossRSA.score_normalized)
analysis_data_GlossVSRSA$model <- as.factor(analysis_data_GlossVSRSA$model)

t.test(score~model, data = analysis_data_GlossVSRSA)
sd(operativesglossRSA$score_normalized)

# difference baseline and baseline RSA
score_data_baselinersa <- data.frame(operativesbaselineadjusted$score_normalized, operativesbaselineRSAadjusted$score_normalized)
analysis_data_baselinersa <- gather(score_data_baselinersa, model, score, operativesbaselineadjusted.score_normalized:operativesbaselineRSAadjusted.score_normalized)
analysis_data_baselinersa$model <- as.factor(analysis_data_baselinersa$model)
t.test(score~model, data=analysis_data_baselinersa)
sd(operativesbaselineRSAadjusted$score_normalized)

sum(studyoperativesgloss$score)
sum(studyoperativesglossRSA$score)
#

#Wilcoxon-Test for ranking:

ranking_data_GlossVSRSA <- data.frame(operativesgloss$rank, operativesglossRSA$rank)

analysis_data_ranking_GlossVSRSA <- gather(ranking_data_GlossVSRSA, model, rank, operativesgloss.rank:operativesglossRSA.rank)
analysis_data_ranking_GlossVSRSA$model <- as.factor(analysis_data_ranking_GlossVSRSA$model)

mean(operativesgloss$rank)
mean(operativesglossRSA$rank)


wilcox.test(rank~model, data = analysis_data_ranking_GlossVSRSA, exact = FALSE, correct = FALSE, conf.int = FALSE)

ranking_data_baselinersa <- data.frame(operativesbaselineadjusted$rank, operativesbaselineRSAadjusted$rank)
analysis_data_ranking_baselinersa <- gather(ranking_data_baselinersa, model, rank, operativesbaselineadjusted.rank:operativesbaselineRSAadjusted.rank)
analysis_data_ranking_baselinersa$model <- as.factor(analysis_data_ranking_baselinersa$model)
wilcox.test(rank~model, data = analysis_data_ranking_baselinersa, exact = FALSE, correct = FALSE, conf.int = FALSE)

#Correlation between participants_no and score_normalized:

#for baseline:

cor.test(operativesbaseline$participants_no, operativesbaseline$score_normalized)
cor.test(operativesgloss$participants_no, operativesgloss$score_normalized)
cor.test(operativesglossRSA$participants_no, operativesglossRSA$score_normalized)


#rep(c(4), times=5)
ggplot(operativesbaselineRSAadjusted, aes(x=n_guesses_given, y=score_normalized)) +
  geom_jitter() +
  geom_smooth(method="lm", se=FALSE)

ggplot(operativesbaselineRSAadjusted, aes(x=additional_given_guesses, y=rank)) +
  geom_jitter() +
  geom_smooth(method="lm", se=FALSE)
cor.test(operativesbaselineRSAadjusted$additional_given_guesses, operativesbaselineRSAadjusted$rank)
ggplot(longaddguess, aes(x=additional_guesses, y=rank, color=model)) +
  geom_jitter() +
  geom_smooth(method="lm", se=FALSE)
cor.test(longaddguess$additional_guesses, longaddguess$rank)
