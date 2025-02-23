#include "PluginMessageBus.h"
#include <QMetaObject>
#include <QDebug>

PluginMessageBus::PluginMessageBus(QObject *parent)
	: QObject(parent)
{
}

PluginMessageBus::~PluginMessageBus()
{
}

void PluginMessageBus::registerPlugin(const QString &pluginName, Plugin* plugin)
{
	if(plugin && !pluginName.isEmpty()){
		this->plugins.insert(pluginName, plugin);
	}else{
		qWarning() << "PluginMessageBus::registerPlugin - Invalid plugin or empty name.";
	}
}

void PluginMessageBus::unregisterPlugin(const QString &pluginName)
{
	this->plugins.remove(pluginName);
}

void PluginMessageBus::sendCommand(const QString &sender,
								   const QString &targetPlugin,
								   const QString &command,
								   const QVariantMap &params)
{
	if(this->plugins.contains(targetPlugin)){
		Plugin* plugin = this->plugins.value(targetPlugin);
		//Invoke the receiveCommand slot on the target plugin asynchronously.
		QMetaObject::invokeMethod(plugin, "receiveCommand", Qt::QueuedConnection,
			Q_ARG(QString, command),
			Q_ARG(QVariantMap, params));
		emit commandSent(sender, targetPlugin, command, params);
	}else{
		qWarning() << "PluginMessageBus::sendCommand - Plugin" << targetPlugin << "not found.";
	}
}

void PluginMessageBus::broadcastCommand(const QString &sender,
										const QString &command,
										const QVariantMap &params)
{
	//Send the command to all registered plugins.
	for(Plugin* plugin : this->plugins){
		QMetaObject::invokeMethod(plugin, "receiveCommand", Qt::QueuedConnection,
			Q_ARG(QString, command),
			Q_ARG(QVariantMap, params));
	}
	emit commandBroadcast(sender, command, params);
}
