#ifndef NETWORK_MANAGER_H
#define NETWORK_MANAGER_H

#include "UDPSocket.h"
#include "TCPSocket.h"
#include <vector>
#include <tchar.h>
#include "VolumeRenderer.h"
#include <future>

class NetworkManager
{
public:
	IPAddress server;
	VolumeRenderer *renderer;
	VolumeDataset *volume;

	UDPSocket udpSocket;
	TCPSocket tcpSocket;

	std::future<void> asyncListener;

	Packet tcpPacket;

	HWND msgWnd;
	
	int udpPort, tcpPort;
	bool connected;

	NetworkManager();
	~NetworkManager();

	void Init(VolumeRenderer *renderer_);
	bool LogIn();
	void AsyncCheckForMessages();
	void CheckForMessages();

	void ReceiveInitialization(Packet &packet);
	void UpdateBlock(Packet &packet);

	bool InitTCPMsgHandler();
	void ReadMessage(WPARAM wParam);
	LRESULT CALLBACK  ProcessMessage(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam);
	static LRESULT CALLBACK MsgWndProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam);
};


#endif