setwd("C:/Users/Luisa/Documents/Uni/SS 2021/Bachelorarbeit/codenames/")
df <- read.csv("results/baseline_ranking_eval.tsv", sep="\t", header=TRUE)
summary(df)
clue_number = str_extract_all(df$clue, " [0-9]")
clue_number
df$clue

ggplot(df, aes(x = c(1:53), y = points)) +
  geom_line()
ggplot(df, aes(x = points)) +
  geom_bar()
sum(df$points)
sum(df$model.score)
