A KWCOCO "bundle_dpath" or "bundle directory" is a directory that a kwcoco file
exists in.

The design philosophy is that the kwcoco file should provide a single path that
provides the system with all information it needs to know about a dataset. My
strong opinion is that ML systems that require the user to configure a path to
the annotations and a path to the images is an error prone anti-pattern that
introduces unnecessary ambiguity into the pipeline.


What you want to do instead is ensure your kwcoco file correctly points at the
paths of the necessary data. When a relative path is specified in a kwcoco file
that always means that the system interprets it with respect to the
bundle_dpath - i.e. the path that the kwcoco file itself exists in.


If your kwcoco file currently points to paths that no longer exist, which can
happen if you move the kwcoco without moving the relative assets that it points
at, then there are a few ways to fix that.

The `kwcoco reroot` CLI tool lets you add or remove a prefix to each path.
There is an equivalent `reroot` API method in the kwcoco.CocoDataset. Reroot
has an `absolute` flag that will force all paths to be rewritten as absolute
paths specific to the machine, which means that you can freely move the kwcoco
file around and it will still be valid as long as the assets don't move and you
are on the same machine.
