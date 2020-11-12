library(tidyverse)
library(tidytext)
library(topicmodels)
library(tm)
library(ldatuning)


arun  <- read_csv("./final_topic_search.csv")
minval <- min(arun$Arun2010)
arun <- arun %>%
    mutate(is_min=if_else(Arun2010 == minval, 1.0, 0.2))
png("./lda_tuning_final.png", width = 6, height=2, units='in', res=300)
ggplot(arun, aes(x=topics, y=Arun2010)) + 
    geom_line() +
    #ylab("KL-Divergence(SVD(TermWeightMatrix), norm(DocTermMatrix))") +
    ylab("KL-Divergence(M1, M2)") +
    xlab("Number of Topics") +
    geom_hline(aes(yintercept=minval), linetype="longdash") +
    geom_point(aes(alpha=is_min), size=1)  +
    jtools::theme_apa(remove.y.gridlines=F, remove.x.gridlines=F) +
    theme(legend.position='none',
          axis.text=element_text(size=8),
          axis.title.x=element_text(size=10),
          axis.title.y=element_text(size=10))

dev.off()

quit(save="no")


fb_data <- readr::read_csv("./temp.txt")
text_ <- fb_data %>%
  rowid_to_column(var="doc_id") %>%
  select(text, doc_id) %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words)

dtm <- text_ %>%
  count(doc_id, word) %>%
  cast_dtm(doc_id, word, n)
print(dtm)

#topic_validate_list <- bsts::GeometricSequence(8, 4, 2)
#topic_validate_list <- c(seq(10,90,10), seq(100, 300, 25))
topic_validate_list <- c(seq(300, 600, 50))
print(topic_validate_list)

validate <- FindTopicsNumber(
  dtm,
  topics=topic_validate_list,
  metrics=c("Arun2010"),
  method="Gibbs",
  mc.cores = 6L,
  control=list(seed=666),
  verbose=TRUE
)
print(validate)
png("./lda_tuning5_large.png", width = 9, height=3, units='in', res=300)
FindTopicsNumber_plot(validate)
dev.off()
readr::write_csv(validate, "./lda_search_results5.csv")
