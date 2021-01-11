TEMPLATE = subdirs

SUBDIRS = \
	octproz_devkit \
	octproz_plugins \
	octproz

octproz_plugins.depends = octproz_devkit
octproz.depends = octproz_devkit


