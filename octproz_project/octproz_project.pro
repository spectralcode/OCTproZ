TEMPLATE = subdirs

SUBDIRS = \
	thirdparty \
	octproz_devkit \
	octproz_plugins \
	octproz

octproz_plugins.depends = octproz_devkit
octproz.depends = octproz_devkit
octproz.depends = thirdparty


