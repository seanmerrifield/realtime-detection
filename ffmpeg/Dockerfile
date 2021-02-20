FROM ubuntu

# Set working directory
WORKDIR /usr/src/app

# Copies everything over to Docker environment
COPY . .

# Update packages
RUN apt-get update
RUN apt-get install -y wget make gcc unzip

# Download file
RUN wget https://github.com/vot/ffbinaries-prebuilt/releases/download/v3.4/ffserver-3.4-linux-64.zip
RUN unzip ffserver-3.4-linux-64.zip


#RUN git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg
#RUN cd ffmpeg && git checkout 2ca65fc7b74444edd51d5803a2c1e05a801a6023
# RUN cd ffmpeg && ./configure
# RUN make -j4

# RUN cd ffmpeg-3.4.8 && ./configure
# #RUN make install
# RUN make -j4

# Expose ports
EXPOSE 3004

CMD ["./ffserver", "-f", "server.conf"]