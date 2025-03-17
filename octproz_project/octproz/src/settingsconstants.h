#ifndef SETTINGS_CONSTANTS_H
#define SETTINGS_CONSTANTS_H

#include <QStandardPaths>

// Settings file paths
#define SETTINGS_DIR QStandardPaths::writableLocation(QStandardPaths::ConfigLocation)
#define SETTINGS_FILE_NAME "settings.ini"
#define SETTINGS_PATH SETTINGS_DIR + "/" + SETTINGS_FILE_NAME
#define GUI_SETTINGS_FILE_NAME "gui_settings.ini"
#define GUI_SETTINGS_PATH SETTINGS_DIR + "/" + GUI_SETTINGS_FILE_NAME
#define SETTINGS_PATH_BACKGROUND_FILE SETTINGS_DIR + "/background.csv"
#define SETTINGS_PATH_RESAMPLING_FILE SETTINGS_DIR + "/resampling.csv"

// Main window settings
#define MAIN_WINDOW_SETTINGS_GROUP "main_window_settings"
#define MAIN_ACTIVE_SYSTEM "main_active_system"
#define MAIN_ACTIVE_EXTENSIONS "main_active_extensions"

// Window geometry constants
#define MAIN_GEOMETRY "main_geometry"
#define MAIN_STATE "main_state"

// Console settings
#define MESSAGE_CONSOLE_BOTTOM "message_console_bottom"
#define MESSAGE_CONSOLE_HEIGHT "message_console_height"

// Dock visibility & geometry
#define DOCK_1DPLOT_VISIBLE "dock_1dplot_visible"
#define DOCK_1DPLOT_GEOMETRY "dock_1dplot_geometry"
#define DOCK_BSCAN_VISIBLE "dock_bscan_visible"
#define DOCK_BSCAN_GEOMETRY "dock_bscan_geometry"
#define DOCK_ENFACEVIEW_VISIBLE "dock_enfaceview_visible"
#define DOCK_ENFACEVIEW_GEOMETRY "dock_enfaceview_geometry"
#define DOCK_VOLUME_VISIBLE "dock_volume_visible"
#define DOCK_VOLUME_GEOMETRY "dock_volume_geometry"

#endif // SETTINGS_CONSTANTS_H
