//PluginMessageBus
//----------------
//This class serves as a central mediator for inter-plugin communication.
//Commands can be sent from any plugin to a specific plugin or broadcast to all registered plugins.
//Each command consists of a sender identifier, a target plugin name, a command string,
//and an optional QVariantMap of parameters (which defaults to an empty map if not provided).

#ifndef PLUGINMESSAGEBUS_H
#define PLUGINMESSAGEBUS_H

#include <QObject>
#include <QMap>
#include <QString>
#include <QVariantMap>
#include "plugin.h"

class PluginMessageBus : public QObject
{
	Q_OBJECT
public:
	explicit PluginMessageBus(QObject *parent = nullptr);
	~PluginMessageBus();

	void registerPlugin(const QString &pluginName, Plugin* plugin);
	void unregisterPlugin(const QString &pluginName);

	//Send a command to a specific plugin by name.
	void sendCommand(const QString &sender,
		const QString &targetPlugin,
		const QString &command,
		const QVariantMap &params = QVariantMap());

	//Broadcast a command to all registered plugins.
	void broadcastCommand(const QString &sender,
		const QString &command,
		const QVariantMap &params = QVariantMap());

signals:
	//Emitted when a command is sent to a specific plugin.
	void commandSent(const QString &sender,
		const QString &targetPlugin,
		const QString &command,
		const QVariantMap &params);

	//Emitted when a command is broadcast to all plugins.
	void commandBroadcast(const QString &sender,
		const QString &command,
		const QVariantMap &params);

private:
	QMap<QString, Plugin*> plugins;
};

#endif // PLUGINMESSAGEBUS_H
