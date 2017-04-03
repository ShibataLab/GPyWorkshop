#Uses official Tensorflow docker for cpu only.
FROM tensorflow/tensorflow
RUN pip install GPy --upgrade
ADD get_files.sh /tmp/get_files.sh
RUN bash /tmp/get_files.sh
RUN rm -f /tmp/get_files.sh
