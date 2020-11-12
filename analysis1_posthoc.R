library(here)
library(tidyverse)
library(car)
library(jtools)
library(emmeans)

regression_results <- read_csv(here("output", "study1_full_results.csv"), 
                               col_types = cols(Representation=col_factor(),
                                                Foundation=col_factor(),
                                                DocThreshold=col_factor(),
                                                Aggregation=col_factor()))
                               

#### Analysis 1 (main analysis) ####

doc_thresh <- "NumberPosts>=10"  # others are MinDocs==1, MinDocs==25
analysis1_df <- regression_results %>%
  filter(DocThreshold==doc_thresh,
         (Aggregation=="non-MIL" & Representation!="BERT") | (Aggregation!="non-MIL" & Representation=="BERT"))

## Single ANOVAs

#library(car)
for(f in c("Care", "Fairness", "Loyalty", "Authority", "Purity")) {
  mod <- lm(R2~Representation, data=analysis1_df %>% filter(Foundation == f))
  print(Anova(mod, type='II'))
  
  values <- analysis1_df %>% filter(Foundation == f)
  print(leveneTest(R2 ~ Representation, data=values))
  print(shapiro.test(values$R2))
  a <- (aov(R2~Representation, data=analysis1_df %>%
                filter(Foundation==f)))
  print(summary(a))
  print(TukeyHSD(a))
}

# Two-way ANOVAs: R2 ~ Representation*Foundation

# in-text references to post hoc comparisons can be made here
mod <- aov(R2~Representation*Foundation, data=analysis1_df)
emm_s.t <- emmeans(mod, pairwise ~ Representation|Foundation)
(emm_plot_df <- plot(emm_s.t, comparisons = F, adjust = "tukey", 
                    horizontal = T, alpha=0.001, plotit=F))

# In SM:
ggplot(emm_plot_df, aes(x=Representation, y=the.emmean)) +
  geom_errorbar(aes(ymin=lower.CL, ymax=upper.CL), width=.5) +
  geom_point() +
  coord_flip() +
  jtools::theme_apa(remove.x.gridlines = F, remove.y.gridlines = F)+
  facet_grid(rows=vars(Foundation)) +
  ylab("Estimated Marginal Mean of Explained Variance") +
  theme(axis.text.x = element_text(size=10),
        axis.text.y = element_text(size=9),
        strip.text = element_text(face='bold', size=10))
ggsave(here("output", "figures", "analysis1", "emms.png"), dpi=300, units='in', 
       width=6.5, height=7.2)


### SM: Aggregation Analysis

agg_analysis_df <- regression_results %>%
  filter(DocThreshold==doc_thresh)
# ANOVAs:
foundations <- c("Care", "Fairness", "Loyalty", "Authority", "Purity")
for(f in foundations) {
  print(f)
  print(car::Anova(lm(R2~Representation*Aggregation, data=agg_analysis_df %>%
                  filter(Foundation==f)), type=2))
}

### SM: PLOT AGGREGATION ANALYSIS
agg_mod <- lm(R2~Representation*Foundation*Aggregation, data=agg_analysis_df)
emm_s.t <- emmeans(agg_mod, ~ Aggregation|Representation)
(emm_plot_agg_df <- plot(emm_s.t, comparisons = T, adjust = "tukey", 
                    horizontal = T, alpha=0.001, plotit=F))
p <- position_dodge(width=0.5)
emm_plot_agg_df
(plotted <- ggplot(emm_plot_agg_df %>%
                     mutate(Representation=factor(Representation, levels=unique(Representation))),
                   aes(x=Representation, color=Aggregation, fill=Aggregation, y=the.emmean)) + 
  geom_errorbar(aes(ymin = lower.CL, ymax = upper.CL), size=1, 
                width=0.2, position=p) +
  #coord_flip() +
  scale_color_manual(values=c("cadetblue", "black")) +
  ylab(expression(paste("Estimated Marginal Mean of ",~R^2))) +
  jtools::theme_apa(remove.y.gridlines = F) +
    theme(legend.position=c(0.8, 0.2)))
ggsave(here("output", "figures", "SM", "aggregation_analysis.png"), dpi=300, units="in", width=6, height=4)

# Sensitivity Analysis (post thresholds)

thresh_df <- regression_results %>%
  filter((Aggregation=="non-MIL" & Representation!="BERT") | (Aggregation!="non-MIL" & Representation=="BERT")) %>%
  dplyr::select(-Aggregation)
thresh_df %>%
  group_by(DocThreshold, Foundation) %>%
  summarize(m=mean(R2), plotrix::std.error(R2))

