library(tidyverse)
library(here)

source(here("./src/R/plots.R"))
source(here("./src/R/models.R"))

IVs <- c("Care", "Fairness", "Loyalty", "Authority", "Purity", 'age', 'gender')
df <- read_tsv(here("data", "features", "public_subjects_dataset.tsv")) %>%
    filter(num_posts >= 10) %>%
    mutate_at(IVs[1:6], funs(c(scale(., scale=T))))

### MFD Analysis
    
mfd2_coefs.no_demo <- run_neg_binomial(df, predictors=IVs[1:5], target_prefix = 'mfd2.count.') 
mfd2_coefs.with_demo <- run_neg_binomial(df, predictors=IVs, target_prefix = 'mfd2.count.') 

(mfd_plot_nodemo <- get_mfd_plot(mfd2_coefs.no_demo))
(mfd_plot_demo <- get_mfd_plot(mfd2_coefs.with_demo))

### LIWC

liwc_coefs.with_demo <- run_neg_binomial(df, predictors=IVs, target_prefix = 'liwc.count.')
liwc_coefs.no_demo <- run_neg_binomial(df, predictors=IVs[1:5], target_prefix = 'liwc.count.')

dictionary_map <- read_csv(here("data", "liwc_map2.csv"))
(liwc_plot_nodemo <- get_liwc_plot(liwc_coefs.no_demo, dictionary_map))
(liwc_plot_demo <- get_liwc_plot(liwc_coefs.with_demo, dictionary_map))

### LDA 

lda_coefs.with_demo <- run_linear(df, IVs, target_prefix = 'lda.')
lda_coefs.no_demo <- run_linear(df, IVs[1:5], target_prefix = 'lda.')

# read word-topic probabilities

(topickeys <- read_tsv(here("output", "topickeys.txt"), 
                      col_names = c("id", "beta", "words")) %>%
    mutate(words=stringr::word(words, 1, 10)))

lda_coefs.with_demo %>%
    mutate(id=gsub('lda\\.', '', id)) %>%
    hablar::convert(hablar::int(id)) %>%
    left_join(topickeys, by = "id") %>%
    filter(estimate > 0, p.value.bonf < 0.05) %>%
    arrange(term, estimate) %>%
    mutate(words=factor(words, levels=unique(words))) %>%
    filter(term %in% c("Care", "Fairness", "Loyalty", "Authority", "Purity")) %>%
        mutate(estimate.text=format(round(estimate*100, 2), digits=2, scientific=F), 
               estimate.text=str_trim(estimate.text),
               estimate.text=if_else(p.value.bonf < 0.05, estimate.text, "")) %>%
    ggplot(aes(term, words)) +
        geom_tile(aes(fill=estimate)) +
        geom_text(aes(label=estimate.text), size=3.5, hjust='center') +
        scale_fill_gradient2(mid='white', low='steelblue', high='red', breaks=pretty_breaks()) +
        theme_bw() + ylab("LDA Probability Outcomes") +
        theme(legend.title = element_blank(),
              axis.title = element_blank(),
              axis.text = element_text(size=9),
              legend.position = "none")
