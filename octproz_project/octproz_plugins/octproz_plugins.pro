TEMPLATE = subdirs

PLUGIN_DIRS = \
	octproz-virtual-oct-system \
	octproz-demo-extension \
	octproz-axial-psf-analyzer-extension \
	octproz-camera-extension \
	octproz-dispersion-estimator-extension \
	octproz-image-statistics-extension \
	octproz-peak-detector-extension \
	octproz-phase-extraction-extension \
	octproz-signal-monitor-extension \
	octproz-socket-stream-extension


for(plugin, PLUGIN_DIRS) {
	plugin_path = $$PWD/$$plugin
	plugin_pro = $$plugin_path/$$plugin.pro

	exists($$plugin_pro) {
		message(Plugin found: $$plugin)
		SUBDIRS += $$plugin
	} else {
		message(Skipping missing plugin: $$plugin)
	}
}

message(Final SUBDIRS: $$SUBDIRS)