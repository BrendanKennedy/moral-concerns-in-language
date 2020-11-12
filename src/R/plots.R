library(scales)

get_liwc_plot <- function(df, dic_name_map) {
  df %>%
    mutate(estimate.text=format(round(estimate, 2), digits=2, scientific=F), 
           estimate.text=str_trim(estimate.text),
           subcategory=gsub('((?:mfd|mfd2|liwc|lda)\\.(?:count|ddr)\\.)', '', id)) %>% 
    left_join(dic_name_map, by = "subcategory") %>%
    mutate(subname=factor(subname, levels=rev(unique(dic_name_map$subname)))) %>%
    drop_na('subname') %>%
    filter(p.value.bonf < 0.01, !term %in% c("age", "genderMale")) -> liwc_plot_df
    p <- ggplot(liwc_plot_df, aes(x=term, y=subname)) +
        geom_tile(aes(fill=statistic)) +
        geom_text(aes(label=estimate.text), size=5, hjust='center') +
            scale_fill_gradient2(low='steelblue', mid='white', high='red', breaks=pretty_breaks()) +
            theme_bw() +
            theme(legend.title = element_blank(),
                  axis.title = element_blank(),
                  axis.text = element_text(size=17),
                  legend.position = "none")
    return(p)
}
get_mfd_plot <- function(df) {
    df %>%
        mutate(estimate.text=format(round(estimate, 2), digits=2, scientific=F), estimate.text=str_trim(estimate.text),
               estimate.text=if_else(p.value.bonf < 0.05, estimate.text, "")) %>%
        mutate(subcategory=gsub('((?:mfd|mfd2|liwc|lda)\\.(?:count|ddr)\\.)', '', id)) %>%
        #left_join(dictionary_map, by = "subcategory") %>%
        mutate(subcategory=recode_factor(subcategory, 
                                         care.virtue="Care",
                                         care.vice="Harm", 
                                         fairness.virtue="Fairness",
                                         fairness.vice="Cheating",
                                         loyalty.virtue="Loyalty",
                                         loyalty.vice="Betrayal",
                                         authority.virtue="Authority",
                                         authority.vice="Subversion", 
                                         sanctity.virtue="Sanctity",
                                         sanctity.vice="Degradation"),
               subcategory=factor(subcategory, levels=rev(levels(subcategory)))) %>%
        filter(term %in% IVs[1:5]) -> df_sub
    ggplot(df_sub, aes(x=term, y=subcategory)) +
        geom_tile(aes(fill=estimate)) +
        geom_text(aes(label=estimate.text), size=5, hjust='center') +
            scale_fill_gradient2(low='steelblue', mid='white', high='red', breaks=pretty_breaks()) +
            theme_bw() +
            theme(legend.title = element_blank(),
                  axis.title = element_blank(),
                  axis.text = element_text(size=17),
                  legend.position = "none") -> p
    return(p)
}