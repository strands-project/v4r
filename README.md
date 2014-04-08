# ACIN V4R mirror

This is the git mirror of ACIN's v4r library for STRANDS.

It uses `git svn` to pull in the latest v4r strands branch from ACIN's svn server:
https://repo.acin.tuwien.ac.at/v4r/stable/strandsv4r

1. `git svn clone https://repo.acin.tuwien.ac.at/v4r/stable/strandsv4r` to check out the svn repo
2. `cd strandsv4r`
3. `git svn rebase` to pull in latest changes from svn 
4. `cd ..`
5. `git pull` to get the latest version from git. This should merge everything nicely. If not, you'll have to deal with any conflicts.
6. Make your changnes. Test it compiles ;-)
7. `git push` to publish the latest version.

Normal users of this repository will usually just run `git pull`, and never push to it.

This repository has been set up following the instructions at http://git-scm.com/book/en/Git-and-Other-Systems-Git-and-Subversion

**It is recommended that usually bugfixes etc for V4R are *not* committed directly to this repository, but rather are fixed in the upstream SVN repository**
