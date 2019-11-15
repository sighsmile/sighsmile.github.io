My blog on github.


----

Note to myself: Latex math formulas with underscores are not properly rendered by Jekyll and mathjax (found this on [stackoverflow](https://stackoverflow.com/questions/49970549/trouble-rendering-some-latex-syntax-in-mathjax-with-jekyll-on-github-pages)).

Solution: wrap each inline formula with in `<span>` tags, and wrap each display formula with `<div>` tags.

Using regex to add tags (in Sublime Text)

- Find: ``(\$\$.+?\$\$)``; Replace: ``<div>$1</div>``
- Find: ``(?<!\$)(\$[^\$]+?\$)(?!\$)``; Replace: ``<span>$1</span>``
