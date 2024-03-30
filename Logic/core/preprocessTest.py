import preprocess
documents = [
            "Scottish warrior William Wallace leads his countrymen in a rebellion to free his homeland from the tyranny of King Edward I of England.",
            "William Wallace is a Scottish rebel whom leads an uprising against the cruel English ruler Edward the Longshanks, who wishes to inherit the crown of Scotland for himself. When he was a young boy, William Wallace&#39;s father and brother, along with many others, lost their lives trying to free Scotland. Once he loses another of his loved ones, William Wallace begins his long quest to make Scotland free once and for all, along with the assistance of Robert the Bruce.<span style=\"display:block\" data-reactroot=\"\">\u2014<a class=\"ipc-link ipc-link--base\" role=\"button\" tabindex=\"0\" aria-disabled=\"false\" href=\"/search/title/?plot_author=Anonymous&amp;view=simple&amp;sort=alpha&amp;ref_=ttpl_pl_2\">Anonymous</a></span>",
            "Tells the story of the legendary thirteenth century Scottish hero named William Wallace. Wallace rallies the Scottish against the English monarch and Edward I after he suffers a personal tragedy by English soldiers. Wallace gathers a group of amateur warriors that is stronger than any English army.<span style=\"display:block\" data-reactroot=\"\">\u2014<a class=\"ipc-link ipc-link--base\" role=\"button\" tabindex=\"0\" aria-disabled=\"false\" href=\"/search/title/?plot_author=Jwelch5742&amp;view=simple&amp;sort=alpha&amp;ref_=ttpl_pl_3\">Jwelch5742</a></span>",
            "Scotland, 1280. With England oppressed by King Edward I, William Wallace, a charismatic Scottish knight of humble descent, leads a righteous campaign to end tyranny. In the commander-in-chief&#39;s gallant quest for freedom during the First War of Scottish Independence, the mighty warrior and gifted strategist risks life and limb to inspire the hopelessly disorganised Scots, oppressed people thirsting for independence. But blood covers the road to liberty.<span style=\"display:block\" data-reactroot=\"\">\u2014<a class=\"ipc-link ipc-link--base\" role=\"button\" tabindex=\"0\" aria-disabled=\"false\" href=\"/search/title/?plot_author=Nick%20Riganas&amp;view=simple&amp;sort=alpha&amp;ref_=ttpl_pl_4\">Nick Riganas</a></span>",
            "In 14th Century Scotland, William Wallace leads his people in a rebellion against the tyranny of the English King, who has given English nobility the &#39;Prima Nocta&#39; - a right to take all new brides for the first night. The Scots are none too pleased with the brutal English invaders, but they lack leadership to fight back. Wallace creates a legend of himself, with his courageous defense of his people and attacks on the English.<span style=\"display:block\" data-reactroot=\"\">\u2014<a class=\"ipc-link ipc-link--base\" role=\"button\" tabindex=\"0\" aria-disabled=\"false\" href=\"/search/title/?plot_author=Rob%20Hartill&amp;view=simple&amp;sort=alpha&amp;ref_=ttpl_pl_5\">Rob Hartill</a></span>"
]

preprocessor = preprocess.Preprocessor(documents)
print(preprocessor.preprocess())

doc3_preprocessed = preprocessor.preprocess_one_text(documents[3])
print(doc3_preprocessed)

doc3 = preprocessor.remove_links(documents[3])
doc3_puncless = preprocessor.remove_punctuations(doc3)
doc3_stopless = preprocessor.remove_stopwords(doc3_puncless)
doc3_normalized = preprocessor.normalize(doc3_stopless)
