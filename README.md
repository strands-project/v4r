# ACIN V4R mirror

This is the git mirror of ACIN's v4r library.

It uses `git svn` to pull in the latest v4r stable branch from ACIN's svn server:
https://repo.acin.tuwien.ac.at/v4r/stable/strands_v4r


## For users

Normal users of this repository just run `git pull`.
Normal uses should never need to push to this repository.


## For maintainers

### Set up the svn git bridge
Run all this NOT from within any of your regular git work spaces.

    git svn clone https://repo.acin.tuwien.ac.at/v4r/stable/strands_v4r
    cd strands_v4r
    git checkout -b hydro-devel
    git remote add origin https://github.com/your-repo/v4r.git
    git push -u origin hydro-devel

### Update
Whenver the svn changed, you have to move these changes to git.

    cd strands_v4r
    git svn rebase
    git push

**It is recommended that usually bugfixes etc for V4R are *not* committed directly to this repository, but rather are fixed in the upstream SVN repository**

---

This repository has been set up following (some of) the instructions at [http://git-scm.com/book/en/Git-and-Other-Systems-Git-and-Subversion]
