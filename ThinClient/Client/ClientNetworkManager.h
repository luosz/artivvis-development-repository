#ifndef NETWORK_MANAGER_H
#define NETWORK_MANAGER_H

#include "UDPSocket.h"
#include "TCPSocket.h"
#include <vector>
#include <tchar.h>

class NetworkManager
{
public:
	int screenWidth, screenHeight;
	unsigned char *pixelBuffer;

	IPAddress server;

	UDPSocket udpSocket;
	TCPSocket tcpSocket;

	Packet tcpPacket;

	HWND msgWnd;
	
	int udpPort, tcpPort;

	NetworkManager();
	~NetworkManager();

	void Init(int screenWidth_, int screenHeight_, unsigned char *buffer);
	bool LogIn();
	bool CheckForMessages();

	void ReceiveInitialization(Packet &packet);
	void UpdateBlock(Packet &packet);

	bool InitTCPMsgHandler();
	void ReadMessage(WPARAM wParam);
	LRESULT CALLBACK  ProcessMessage(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam);
	static LRESULT CALLBACK MsgWndProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam);
};


#endif