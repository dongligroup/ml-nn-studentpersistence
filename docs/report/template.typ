#let article(
  // Article's Title
  title: "Article Title",
  
  // A dictionary of authors.
  // Dictionary keys are authors' names.
  // Dictionary values are meta data of every author, including
  // label(s) of affiliation(s), email, contact address,
  // or a self-defined name (to avoid name conflicts).
  // Once the email or address exists, the author(s) will be labelled
  // as the corresponding author(s), and their address will show in footnotes.
  // 
  // Example:
  // (
  //   "Auther Name": (
  //     "affiliation": "affil-1",
  //     "email": "author.name@example.com", // Optional
  //     "address": "Mail address",  // Optional
  //     "name": "Alias Name" // Optional
  //   )
  // )
  authors: (),

  // A dictionary of affiliation.
  // Dictionary keys are affiliations' labels.
  // These labels show be constent with those used in authors' meta data.
  // Dictionary values are addresses of every affiliation.
  //
  // Example:
  // (
  //   "affil-1": "Institudion Name, University Name, Road, Post Code, Country"
  // )
  affiliations: (),

  // The paper's abstract.
  abstract: [],

  // The paper's keywords.
  keywords: (),

  // The path to a bibliography file if you want to cite some external
  // works.
  bib: none,

  // Paper's content
  body
) = {
  // Set document properties
  set document(title: title, author: authors.keys())
  set page(numbering: "1", number-align: center,paper: "us-letter")
  set text(font: ( "New Computer Modern", "Times New Roman"), lang: "en")
  show footnote.entry: it => [
    #set par(hanging-indent: 0.7em)
    #it.note.numbering #it.note.body
  ]

  // Title block
  align(center)[
    #block(text(font: "Palatino Linotype", size: 1.52em, weight: "bold", smallcaps(title)))
  ]

  v(1em)

  // Authors and affiliations
  align(center)[

    // Restore affiliations' keys for looking up later
    // to show superscript labels of affiliations for each author.
    #let inst_keys = affiliations.keys()

    // Authors' block
    #block([
      // Process the text for each author one by one
      #for (ai, au) in authors.keys().enumerate() {
        let au_meta = authors.at(au)
        // Don't put comma before the first author
        if ai != 0 {
          text([, ])
        }
        // Write auther's name
        if au_meta.keys().contains("name") {
          text([#au_meta.name])
        } else {
          text([#au])
        }

        // Get labels of author's affiliation
        let au_inst_id = au_meta.affiliation
        let au_inst_primary = ""
        // Test whether the author belongs to multiple affiliations
        if type(au_inst_id) == "array" {
          // If the author belongs to multiple affiliations,
          // record the first affiliation as the primary affiliation,
          au_inst_primary = affiliations.at(au_inst_id.first())
          // and convert each affiliation's label to index
          let au_inst_index = au_inst_id.map(id => inst_keys.position(key => key == id) + 1)
          // Output affiliation
          super([#(au_inst_index.map(id => [#id]).join([,]))])
        } else if (type(au_inst_id) == "string") {
          // If the author belongs to only one affiliation,
          // set this as the primary affiliation
          au_inst_primary = affiliations.at(au_inst_id)
          // convert the affiliation's label to index
          let au_inst_index = inst_keys.position(key => key == au_inst_id) + 1
          // Output affiliation
          super([#au_inst_index])
        }

        // Corresponding author
        if au_meta.keys().contains("email") or au_meta.keys().contains("address") {
          footnote(numbering: "*")[
            Corresponding author. Address:
            #if not au_meta.keys().contains("address") or au_meta.address == "" {
              [#au_inst_primary.]
            }
            #if au_meta.keys().contains("email") {
              [Email: #underline(au_meta.email).]
            }
          ]
        }
      }
    ])

    #v(-0.2em)

    // Affiliation block
    #block([
      #set par(leading: 0.4em)
      #for (ik, key) in inst_keys.enumerate() {
        text(size: 0.8em, [#super([#(ik+1)]) #(affiliations.at(key))])
        linebreak()
      }
    ])
  ]

  // Abstract and keyword block
  if abstract != [] {
    v(1em)

    block([
      #set par(first-line-indent: 0em, justify: true)
      #align(center)[#heading(smallcaps[Abstract])]
      #abstract
  
      #if keywords.len() > 0 {
        text(weight: "bold", [Key words: ])
        text([#keywords.join([; ]).])
      }
    ])

    v(1em)
  }

  // Display contents

  show heading.where(level: 1): it => block(above: 1.5em, below: 1.5em)[
    #set pad(bottom: 2em, top: 1em)
    #smallcaps(it.body)
  ]

  show: rest => columns(2, rest)

  set par(first-line-indent: 0em, justify: true)
  
  body

  // Display bibliography.
  if bib != none {
    show bibliography: set text(1em)
    show bibliography: set par(first-line-indent: 0em)
    bibliography(bib, title: [References], style: "ieee")
  }

  
}