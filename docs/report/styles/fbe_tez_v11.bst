% Imitation of the CMPE 492 report template style (original fbe_tez.sty not available)
% Numbered, citation order. Entry form:
%   Surname1, N. and N. Surname2, ``Title'', {\em Venue}, Vol. X, No. Y, pp. a--b, Publisher, Year.

ENTRY
  { address author booktitle edition editor howpublished institution journal
    note number organization pages publisher school title type url volume year }
  {}
  { label }

INTEGERS { output.state before.all mid.sentence nameptr namesleft numnames entries }
STRINGS { s t }

FUNCTION {init.state.consts}
{ #0 'before.all :=
  #1 'mid.sentence :=
}

FUNCTION {output.nonnull}
{ 's :=
  output.state mid.sentence =
    { ", " * write$ }
    { write$ mid.sentence 'output.state := }
  if$
  s
}

FUNCTION {output}
{ duplicate$ empty$ 'pop$ 'output.nonnull if$ }

FUNCTION {emphasize}
{ duplicate$ empty$ { pop$ "" } { "{\em " swap$ * "}" * } if$ }

FUNCTION {quote}
{ duplicate$ empty$ { pop$ "" } { "``" swap$ * "''" * } if$ }

FUNCTION {output.bibitem}
{ newline$
  "\bibitem{" write$
  cite$ write$
  "}" write$
  newline$
  ""
  before.all 'output.state :=
}

FUNCTION {fin.entry}
{ duplicate$ empty$ 'pop$ { "." * write$ } if$
  newline$
}

FUNCTION {format.names}
{ 's :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames 'namesleft :=
  ""
  { namesleft #0 > }
  { nameptr #1 =
      { s nameptr "{vv~}{ll}{, f.}{, jj}" format.name$ 't := }
      { s nameptr "{f.~}{vv~}{ll}{, jj}" format.name$ 't := }
    if$
    nameptr #1 >
      { namesleft #1 >
          { ", " * t * }
          { t "others" =
              { " et al." * }
              { " and " * t * }
            if$
          }
        if$
      }
      { t * }
    if$
    nameptr #1 + 'nameptr :=
    namesleft #1 - 'namesleft :=
  }
  while$
}

FUNCTION {format.authors}
{ author empty$ { "" } { author format.names } if$ }

FUNCTION {format.vol}
{ volume empty$ { "" } { "Vol.~" volume * } if$ }

FUNCTION {format.num}
{ number empty$ { "" } { "No.~" number * } if$ }

FUNCTION {format.pages}
{ pages empty$ { "" } { "pp.~" pages * } if$ }

FUNCTION {format.url}
{ url empty$ { "" } { "\url{" url * "}" * } if$ }

FUNCTION {article}
{ output.bibitem
  format.authors output
  title quote output
  journal emphasize output
  format.vol output
  format.num output
  format.pages output
  year output
  note output
  format.url output
  fin.entry
}

FUNCTION {inproceedings}
{ output.bibitem
  format.authors output
  title quote output
  booktitle emphasize output
  format.vol output
  format.pages output
  publisher output
  year output
  note output
  format.url output
  fin.entry
}

FUNCTION {conference} { inproceedings }

FUNCTION {book}
{ output.bibitem
  format.authors output
  title emphasize output
  publisher output
  address output
  year output
  note output
  fin.entry
}

FUNCTION {techreport}
{ output.bibitem
  format.authors output
  title emphasize output
  type empty$ { "Technical Report" } { type } if$ output
  institution output
  address output
  year output
  note output
  format.url output
  fin.entry
}

FUNCTION {misc}
{ output.bibitem
  format.authors output
  title quote output
  howpublished output
  year output
  note output
  format.url output
  fin.entry
}

FUNCTION {default.type} { misc }

READ

FUNCTION {initialize} { #0 'entries := }
EXECUTE {initialize}
FUNCTION {count} { entries #1 + 'entries := }
ITERATE {count}

FUNCTION {begin.bib}
{ "\begin{thebibliography}{" entries int.to.str$ * "}" * write$ newline$ }

EXECUTE {begin.bib}
EXECUTE {init.state.consts}
ITERATE {call.type$}

FUNCTION {end.bib}
{ newline$ "\end{thebibliography}" write$ newline$ }

EXECUTE {end.bib}
EXECUTE {init.state.consts}
