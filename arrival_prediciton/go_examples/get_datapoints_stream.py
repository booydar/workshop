# proto command:
# python -m grpc_tools.protoc -I /home/ayd98/Desktop/_projects/arrival_prediction/go_examples --python_out=. --grpc_python_out=. Api.proto

import grpc

import Api_pb2 as api
import Api_pb2_grpc as api_grpc
import pandas as pd
# import route_guide_resources

TIME_FORMAT = "2006-01-02 15:04:05"


def get_data_stream(stub):

    timeFrom = pd.to_datetime("2020-09-13 21:00:00", format='%Y-%m-%d %H:%M:%S')
    timeTo = pd.to_datetime("2020-09-14 21:00:00", format='%Y-%m-%d %H:%M:%S')

    rangeStreamRequest = api.ObjectsDataRangeRequest(
		Filter= api.DataFilter(
			DateFrom= 			int(timeFrom.timestamp()),
			DateTo= 			int(timeTo.timestamp()),
            Subsystem= 			"kiutr", # для мусоровозов - garbade, на тестовом стенде данные по garbage отсутствуют 
			ExcludeDeviceCode= 	["10033473", "404957","500459"], # пример исключения уже обработанных блоков
			DeviceCode=			["10033473","404957"],  # дополнительные коды БНСО
			StateNumber= 		["Н040РА195"],  # дополнительные госномера
			# Subsystem= 			[]string{"kiutr"}, // для мусоровозов - garbade, на тестовом стенде данные по garbage отсутствуют 
			# ExcludeDeviceCode= 	[]string{"10033473","404957","500459"}, // пример исключения уже обработанных блоков
			# DeviceCode=			[]string{"10033473","404957"},  // дополнительные коды БНСО
			# StateNumber= 		[]string{"Н040РА195"},  // дополнительные госномера
        ),
		Fields = api.FieldsToggle(
			Position=True # запрашивает только навигационную информацию
        )
    )

    stream = stub.GetObjectsDataRangeAsStream(rangeStreamRequest)

    return stream
    

    
def run():
    # endpoint_address = 'rnis-tm.t1-group.ru:18082'
    localhost = 'localhost:50051'
    with grpc.insecure_channel(localhost) as channel:
        print("------------Create Stub--------------")
        stub = api_grpc.APIStub(channel)

        print("------------Get DataStream-----------")
        stream = get_data_stream(stub)

        print("------ Retreive data from stream-----")
        i = 0
        for object_data in stream:
            i += 1
            if i == 5:
                break

            device_code = object_data.DeviceCode
            # state_number = object_data.StateNumber
            data_point = object_data.DataPoint

            device_time = data_point.DeviceTime
            gps_data = data_point.Position
            state = data_point.ObjectState
            # received_time = data_point.ReceivedTime
            accelerations = data_point.Accelerations
            # fuel_spent = data_point.FuelSpent

            longitude = gps_data.Longitude
            latitude = gps_data.Latitude
            altitude = gps_data.Altitude
            course = gps_data.Course
            # satellites = gps_data.Satellites
            speed = gps_data.Speed
            valid = gps_data.Valid
            hdop = gps_data.HDOP

            print('datapoint ', i)
            print('device_code: ', device_code)
            print('device_time: ', device_time)
            print('state: ', state)
            print('longitude: ', longitude)
            print('latitude: ', latitude)
            print('altitude: ', altitude)
            print('course: ', course)
            print('speed: ', speed)
            print('accelerations: ', accelerations)
            print('valid: ', valid)
            print('hdop: ', hdop)
            print()



def pack_request_to_df(filename, max_records):
    # endpoint_address = 'rnis-tm.t1-group.ru:18082'
    localhost = 'localhost:50051'

    records_df = pd.DataFrame(columns=['device_code', 
                                'state_number',
                                'course',
                                'device_time',
                                'state',
                                'longitude',
                                'latitude',
                                'altitude',
                                'speed', 
                                'accelerations',
                                'fuel_spent',
                                'satellites'
                                'valid',
                                'hdop'])

     
    with grpc.insecure_channel(localhost) as channel:
        print("------------Create Stub--------------")
        stub = api_grpc.APIStub(channel)

        print("------------Get DataStream-----------")
        stream = get_data_stream(stub)

        print("------ Retreive data from stream-----")

        i = 0
        for object_data in stream:
            i += 1
            if i == max_records:
                break

        data_point = object_data.DataPoint
        gps_data = data_point.Position
        record = {
            'device_code': object_data.DeviceCode,
            'state_number': object_data.StateNumber,
            'device_time': data_point.DeviceTime,            
            'state': data_point.ObjectState,
            'received_time': data_point.ReceivedTime,
            'accelerations': data_point.Accelerations,
            'fuel_spent': data_point.FuelSpent,
            'longitude': gps_data.Longitude,
            'latitude': gps_data.Latitude,
            'altitude': gps_data.Altitude,
            'course': gps_data.Course,
            'satellites': gps_data.Satellites,
            'speed': gps_data.Speed,
            'valid': gps_data.Valid,
            'hdop': gps_data.HDOP,
        }

        records_df.append(record, ignore_index=True)

if __name__ == '__main__':
    
    run()
    
    # pack_request_to_df('log.csv', max_records=10_000)