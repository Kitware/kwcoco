Gotchas
-------

These are the gotchas that people have encounted when using the library. We
will document, and then maybe fix them.

* If you request a channel with ``imdelay`` that does not exist, it will return
  NaN. This is because imdelay gives you the concept of an "infinite canvas"
  and you can crop on any point on it, but if you don't have an observation, it
  will return nan.
