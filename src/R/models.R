
run_linear <- function(dataframe, predictors, target_prefix) {
  level_names <- predictors[1:5]
    K <- length(predictors)
    if(K > 5) {
        for(i in 6:K) {
            iv_name <- predictors[i]
            if(iv_name == 'age') level_names <- c(level_names, "age")
            if(iv_name == 'gender') level_names <- c(level_names, "genderMale")
        }
    }
    coef_df <- dataframe %>%
        dplyr::select(starts_with(target_prefix), all_of(predictors), 'subject_id') %>%
        gather("id", "lda_prob", starts_with(target_prefix)) %>%
        nest_by(id) %>%
        mutate(fit_model = list(lm(as.formula(paste("lda_prob ~ ", paste(predictors, collapse=" + "), collapse=" ")),
                                         data=data)))  %>%
        summarise(broom::tidy(fit_model)) %>%
        filter(term != "(Intercept)") %>%
        mutate(p.value.bonf=p.adjust(p.value, method='bonferroni')) %>%
        ungroup(id) %>%
        mutate(term = factor(term, levels = level_names))
    
    return(coef_df)
}

run_neg_binomial <- function(dataframe, predictors, target_prefix) {
    count_offset_term <- "num_words"
    level_names <- predictors[1:5]
    K <- length(predictors)
    if(K > 5) {
        for(i in 6:K) {
            iv_name <- predictors[i]
            if(iv_name == 'age') level_names <- c(level_names, "age")
            if(iv_name == 'gender') level_names <- c(level_names, "genderMale")
        }
    }
    coef_df <- dataframe %>%
        dplyr::select(starts_with(target_prefix), all_of(predictors), all_of(c(count_offset_term)), 'subject_id') %>%
        gather("id", "count", starts_with(target_prefix)) %>%
        nest_by(id) %>%
        mutate(fit_model = list(MASS::glm.nb(as.formula(paste("count ~ offset(log(num_words)) + ", 
                                               paste(predictors, collapse=" + "), collapse=" ")),
                                         data=data)))  %>% 
        summarise(broom::tidy(fit_model)) %>%
        filter(term != '(Intercept)') %>%
        mutate(p.value.bonf=p.adjust(p.value, method='bonferroni')) %>%
        ungroup(id) %>%
        mutate(term = factor(term, levels = level_names))
    return(coef_df)
}
