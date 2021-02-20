# specify the node base image with your desired version node:<version>
FROM node:14.15.5-alpine3.10

# # copy project
ADD . /usr/src/app/
WORKDIR /usr/src/app/

RUN npm install

# replace this with your application's default port
EXPOSE 3001
EXPOSE 3002

CMD ["node", "./node-server/server.js"]