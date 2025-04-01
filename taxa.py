import polars as pl

taxa_df = pl.read_csv('taxa.csv')
common_names_df = pl.read_csv('VernacularNames-english.csv')

taxa_df = taxa_df.filter(pl.col("taxonRank") == "species")
species_df = taxa_df.join(common_names_df, on="id", how="inner")

species_df.write_csv("species_with_taxonomic_info.csv")