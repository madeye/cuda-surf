//
//  Hello World client in C++
//  Connects REQ socket to tcp://localhost:5555
//  Sends "Hello" to server, expects "World" back
//
#include "zmq.hpp"
#include <string>
#include <iostream>

int main (int args, char** argv)
{
    if (args < 4) return -1;

    char text[512];
    char url[128];

    //  Prepare our context and socket
    zmq::context_t context (1);
    zmq::socket_t socket (context, ZMQ_REQ);

    sprintf(url, "tcp://127.0.0.1:%s", argv[1]);
    std::cout << "Connecting to voctree server..." << std::endl;
    socket.connect (url);

    sprintf(text, "%s\n%s", argv[2], argv[3]);
    zmq::message_t request (strlen(text) + 1);
    memcpy ((void *) request.data(), text, strlen(text) + 1);
    printf("%s\n", (char *) request.data());

    socket.send (request);

    //  Get the reply.
    zmq::message_t reply;
    socket.recv (&reply);
    std::cout << "Received " << ((char *) reply.data()) << std::endl;

    return 0;
}
