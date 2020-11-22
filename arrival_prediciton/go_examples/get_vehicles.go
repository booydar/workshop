package main

import (
	"google.golang.org/grpc"
	"context"
	"os"
	"git-02.t1-group.ru/contracts/proto-go/api"
	log "github.com/sirupsen/logrus"
)

func getGrpcClient(URL string) (api.APIClient, error){
	log.Infof("# Соединение c gRPC сервисом...")
	conn, err := grpc.Dial(URL, grpc.WithInsecure())
	if err != nil {
		return nil, err
	}
	cl := api.NewAPIClient(conn)
	return cl, nil
}

func printVehicle(vehicle *api.Vehicle) {
	log.Infof("  Subsystem: %s \t DeviceCode: %s", vehicle.Subsystem, vehicle.DeviceCode)
	log.Infof("\tStateNumber: %s", vehicle.StateNumber)
	log.Infof("\tVehicleMark: %s", vehicle.VehicleMark)
	log.Infof("\tVehicleModel: %s", vehicle.VehicleModel)
	log.Infof("\tUnitINN: %s", vehicle.UnitINN)
}

func getVehicles(client api.APIClient, subsystems []string) {
	log.Infof("# Запрос...")
	request := &api.GetVehiclesRequest{
		Subsystems: subsystems,
	}

	result, err := client.GetVehicles(context.Background(), request)
	if err != nil {
		log.Errorf("failed to requsest: %v", err)
		return
	}

	if len(result.Vehicles) > 0 {
		log.Infof("Response items count %d with subsystems %v (displaying first item)", len(result.Vehicles), subsystems)
		printVehicle(result.Vehicles[0])
	}
}

func main() {
	client, err := getGrpcClient(os.Getenv("GATEWAY_ADDR"))

	if err != nil {
		log.Errorf("Ошибка соединения c gRPC сервисом: %v", err)
		return
	}
	getVehicles(client, []string{})
	getVehicles(client, []string{"kiutr"})
}
