package main

import (
	"google.golang.org/grpc"
	"time"
	"io"
	"context"
	// "externalapi/api"
	"git-02.t1-group.ru/contracts/proto-go/api"
	log "github.com/sirupsen/logrus"
)

const TIME_FORMAT = "2006-01-02 15:04:05"


func main() {

	log.Infof("# Соединение c gRPC сервисом...")
	conn, err := grpc.Dial("rnis-tm.t1-group.ru:18082", grpc.WithInsecure())
	if err != nil {
		log.Errorf("# Ошибка соединения c gRPC сервисом: %v", err)
	}

	client := api.NewAPIClient(conn)

	log.Infof("# Запрос диапазона телематики в формате потока...")

	timeFrom, _	:= time.Parse(TIME_FORMAT, "2020-09-13 21:00:00")
	timeTo, _ 	:= time.Parse(TIME_FORMAT, "2020-09-14 21:00:00")

	rangeStreamRequest := &api.ObjectsDataRangeRequest{
		Filter: &api.DataFilter{
			DateFrom: 			timeFrom.Unix(),
			DateTo: 			timeTo.Unix(),
			Subsystem: 			[]string{"kiutr"}, // для мусоровозов - garbade, на тестовом стенде данные по garbage отсутствуют 
			ExcludeDeviceCode: 	[]string{"10033473","404957","500459"}, // пример исключения уже обработанных блоков
			DeviceCode:			[]string{"10033473","404957"},  // дополнительные коды БНСО
			StateNumber: 		[]string{"Н040РА195"},  // дополнительные госномера
		},
		Fields: &api.FieldsToggle{
			Position: true, // заправшивает только навигационную информацию
		},
	}

	rangeStream, err := client.GetObjectsDataRangeAsStream(context.Background(), rangeStreamRequest)
	if err != nil {
		log.Error("# Ошибка получения потока: ", err)
	}
	for {
	    object, err := rangeStream.Recv()
	    if err != nil {
			if err == io.EOF {
		    	log.Error("# Поток закрыт, все данные переданы ", err)
		        break
	    	} else {
		        log.Fatalf("# Ошибка получения данных: %v", err)
	    	}
	    }

		log.Infof("ObjectID: %s \t StateNumber: %s", object.DeviceCode, object.StateNumber)
		// object.Data содержит в себе все запрошенные телематические точки по устройству
		for _, point := range object.Data {
			log.Infof("\t DeviceTime: %s \tLongitude: %.8f \t Latitude: %.8f", time.Unix(point.DeviceTime, 0), point.Position.Longitude, point.Position.Latitude)
		}
	}
}